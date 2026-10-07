"""E(3)-equivariant k-step models.

Both models share a PaiNN-style backbone (Schütt et al., ICML 2021) operating on the
fully connected intra-molecular graph. Every node carries invariant scalar features
``s`` and equivariant vector features ``V``; vector inputs (velocity, and for the flow
model the noisy Δx/Δv) only ever enter through channel-mixing linear maps without bias
and through dot products, and vector outputs are built from ``V`` alone. The
predictions are therefore exactly rotation/reflection equivariant, translation
invariant and permutation equivariant – no augmentation needed.

Two heads are provided:

``kind: mean``  Deterministic regression of (Δx, Δv) = (x_{t+k} - x_t, v_{t+k} - v_t).
                With Langevin training data this learns the *conditional mean*.
``kind: flow``  Conditional flow matching (Lipman et al. 2023). The network learns a
                velocity field that transports isotropic Gaussian noise to samples of
                p(Δx, Δv | x_t, v_t), so rollouts are stochastic like the underlying
                Langevin dynamics instead of collapsing to the mean.

All quantities are normalised per element (tables fitted on the training set and
stored as buffers, so checkpoints are self-contained). Predictions are projected onto
the zero-COM / zero-momentum subspace, so linear momentum is conserved exactly.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .common import LOGGER

Batch = Dict[str, torch.Tensor]


@dataclass
class ModelConfig:
    kind: str = "flow"            # "mean" | "flow"
    hidden_dim: int = 128
    num_layers: int = 4
    num_rbf: int = 32
    cutoff_nm: float = 1.2        # interactions vanish smoothly (cosine envelope) at this distance
    max_atomic_number: int = 100
    msg_norm: float = 8.0         # divides summed messages (≈ typical neighbour count)
    time_embed_dim: int = 32      # flow only
    flow_steps: int = 8           # default ODE steps at sampling time
    flow_solver: str = "heun"     # "euler" | "heun"
    max_disp_nm: float = 0.0      # optional equivariant soft clamp on |Δx| per atom (0 = off)
    max_dvel_nm_per_ps: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            LOGGER.warning("Ignoring unknown model config keys: %s", ", ".join(unknown))
        return cls(**{k: v for k, v in data.items() if k in names})


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def full_graph(batch: torch.Tensor) -> torch.Tensor:
    """All ordered pairs (i, j), i != j, within each graph. ``batch`` must be sorted.

    Fully vectorised (no Python loop over graphs). Returns ``(2, E)`` with
    ``edge_index[0]`` = receiver i and ``edge_index[1]`` = sender j.
    """
    device = batch.device
    n = batch.numel()
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    counts = torch.bincount(batch)
    ptr = torch.cumsum(counts, 0) - counts                      # first node of each graph
    deg = (counts[batch] - 1).clamp(min=0)                      # neighbours per node
    row = torch.repeat_interleave(torch.arange(n, device=device), deg)
    start = torch.cumsum(deg, 0) - deg
    k = torch.arange(row.numel(), device=device) - start[row]   # 0..deg-1 for each receiver
    local_i = torch.arange(n, device=device) - ptr[batch]
    col_local = k + (k >= local_i[row]).long()                  # skip self
    col = ptr[batch[row]] + col_local
    return torch.stack([row, col], dim=0)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


def project_zero_com(y: torch.Tensor, weights: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Orthogonal projection removing the weighted per-graph sum: Σ_i w_i y_i = 0.

    ``y`` is ``(N, 3)``, ``weights`` is ``(N,)``. Equivariant because w is a scalar.
    """
    w = weights.unsqueeze(-1)
    num = scatter_sum(w * y, batch, num_graphs)                  # (G, 3)
    den = scatter_sum(w * w, batch, num_graphs).clamp(min=1e-12)  # (G, 1)
    return y - w * (num / den)[batch]


def soft_clamp_norm(vec: torch.Tensor, limit: float) -> torch.Tensor:
    """Smoothly cap per-row vector norms at ``limit`` (rotation equivariant)."""
    if limit <= 0:
        return vec
    norm = vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return vec * (limit * torch.tanh(norm / limit) / norm)


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.gamma = (num_rbf / cutoff) ** 2 / 2.0
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff)
        return rbf, envelope


def _mix_channels(linear: nn.Linear, vec: torch.Tensor) -> torch.Tensor:
    """Apply a bias-free Linear over the channel axis of ``(N, C, 3)`` vectors."""
    return linear(vec.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# PaiNN blocks
# ---------------------------------------------------------------------------

class PaiNNMessage(nn.Module):
    def __init__(self, dim: int, num_rbf: int) -> None:
        super().__init__()
        self.dim = dim
        self.phi = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.filter = nn.Linear(num_rbf, 3 * dim)

    def forward(self, s, V, edge_index, rbf, envelope, unit, norm: float):
        i, j = edge_index
        x = self.phi(s)[j] * self.filter(rbf) * envelope.unsqueeze(-1)
        a, b, c = torch.split(x, self.dim, dim=-1)
        ds = scatter_sum(a, i, s.size(0))
        dV = scatter_sum(b.unsqueeze(-1) * V[j] + c.unsqueeze(-1) * unit.unsqueeze(1), i, s.size(0))
        return s + ds / norm, V + dV / norm


class PaiNNUpdate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.U = nn.Linear(dim, dim, bias=False)
        self.W = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, s, V):
        UV = _mix_channels(self.U, V)
        WV = _mix_channels(self.W, V)
        wv_norm = torch.sqrt((WV ** 2).sum(-1) + 1e-8)
        a = self.mlp(torch.cat([s, wv_norm], dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.dim, dim=-1)
        V = V + a_vv.unsqueeze(-1) * UV
        s = s + a_sv * (UV * WV).sum(-1) + a_ss
        return self.norm(s), V


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(1000.0) / max(half - 1, 1)))
    args = t.unsqueeze(-1) * freqs * 2 * math.pi
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EquivariantBackbone(nn.Module):
    """Maps (positions, per-atom input vectors, scalars) to two output vectors per atom."""

    def __init__(self, cfg: ModelConfig, num_in_vectors: int, extra_scalar_dim: int = 0) -> None:
        super().__init__()
        F = cfg.hidden_dim
        self.cfg = cfg
        self.num_in = num_in_vectors
        n_gram = num_in_vectors * (num_in_vectors + 1) // 2
        self.embed = nn.Embedding(cfg.max_atomic_number + 1, F)
        self.scalar_in = nn.Sequential(nn.Linear(F + n_gram + extra_scalar_dim, F), nn.SiLU(), nn.Linear(F, F))
        self.vec_in = nn.Linear(num_in_vectors, F, bias=False)
        self.rbf = GaussianRBF(cfg.num_rbf, cfg.cutoff_nm)
        self.messages = nn.ModuleList([PaiNNMessage(F, cfg.num_rbf) for _ in range(cfg.num_layers)])
        self.updates = nn.ModuleList([PaiNNUpdate(F) for _ in range(cfg.num_layers)])
        self.gate = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, F))
        self.vec_out = nn.Linear(F, 2, bias=False)
        tri = torch.triu_indices(num_in_vectors, num_in_vectors)
        self.register_buffer("tri_i", tri[0], persistent=False)
        self.register_buffer("tri_j", tri[1], persistent=False)

    def forward(self, pos, z, batch, in_vectors, extra_scalars=None, edge_index=None):
        """``in_vectors``: ``(N, C, 3)``. Returns ``(N, 2, 3)``."""
        if edge_index is None:
            edge_index = full_graph(batch)
        i, j = edge_index
        rel = pos[j] - pos[i]
        dist = rel.norm(dim=-1)
        # Pairs beyond the cutoff are switched off by the envelope rather than removed,
        # which keeps tensor shapes static (CUDA-graph friendly) at small extra cost.
        unit = rel / dist.clamp(min=1e-9).unsqueeze(-1)
        rbf, envelope = self.rbf(dist)

        gram = torch.einsum("nci,ndi->ncd", in_vectors, in_vectors)[:, self.tri_i, self.tri_j]
        scal = [self.embed(z), gram]
        if extra_scalars is not None:
            scal.append(extra_scalars)
        s = self.scalar_in(torch.cat(scal, dim=-1))
        V = _mix_channels(self.vec_in, in_vectors)

        for msg, upd in zip(self.messages, self.updates):
            s, V = msg(s, V, edge_index, rbf, envelope, unit, self.cfg.msg_norm)
            s, V = upd(s, V)
        return _mix_channels(self.vec_out, self.gate(s).unsqueeze(-1) * V)


# ---------------------------------------------------------------------------
# k-step model (mean or flow)
# ---------------------------------------------------------------------------

class KStepModel(nn.Module):
    """Predicts the k-step update (Δx, Δv) for a batch of COM-centred molecules.

    Batch keys (all torch tensors, concatenated over molecules):
      ``x_t`` (N,3) nm, ``v_t`` (N,3) nm/ps, ``atom_types`` (N,) long,
      ``masses`` (N,), ``batch`` (N,) graph index (sorted);
      for training additionally ``x_tk``, ``v_tk``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.kind not in ("mean", "flow"):
            raise ValueError(f"Unknown model kind {cfg.kind!r}; expected 'mean' or 'flow'")
        self.cfg = cfg
        nz = cfg.max_atomic_number + 1
        # Per-element normalisation tables (filled by set_normalization before training)
        for name in ("vel_scale", "dpos_scale", "dvel_scale"):
            self.register_buffer(name, torch.ones(nz))
        if cfg.kind == "flow":
            self.backbone = EquivariantBackbone(cfg, num_in_vectors=3, extra_scalar_dim=cfg.time_embed_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim), nn.SiLU(), nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim)
            )
        else:
            self.backbone = EquivariantBackbone(cfg, num_in_vectors=1)

    # -- normalisation -------------------------------------------------------
    @torch.no_grad()
    def set_normalization(self, tables: Dict[str, torch.Tensor]) -> None:
        for name, table in tables.items():
            getattr(self, name).copy_(torch.as_tensor(table, dtype=torch.float32))

    def _scales(self, z: torch.Tensor):
        return self.vel_scale[z].unsqueeze(-1), self.dpos_scale[z].unsqueeze(-1), self.dvel_scale[z].unsqueeze(-1)

    @staticmethod
    def _num_graphs(batch: Batch) -> int:
        if "num_graphs" in batch:
            return int(batch["num_graphs"])
        return int(batch["batch"].max().item()) + 1 if batch["batch"].numel() else 0

    def _project(self, yx, yv, batch: Batch, sx, sv, G):
        m = batch["masses"]
        yx = project_zero_com(yx, m * sx.squeeze(-1), batch["batch"], G)
        yv = project_zero_com(yv, m * sv.squeeze(-1), batch["batch"], G)
        return yx, yv

    def _field(self, batch: Batch, yx, yv, tau, edge_index=None):
        """Flow velocity field in normalised space (or mean prediction if kind == mean)."""
        z = batch["atom_types"]
        vin = batch["v_t"] / self.vel_scale[z].unsqueeze(-1)
        if self.cfg.kind == "flow":
            vecs = torch.stack([vin, yx, yv], dim=1)
            temb = self.time_mlp(sinusoidal_embedding(tau, self.cfg.time_embed_dim))
            out = self.backbone(batch["x_t"], z, batch["batch"], vecs, temb, edge_index)
        else:
            out = self.backbone(batch["x_t"], z, batch["batch"], vin.unsqueeze(1), None, edge_index)
        return out[:, 0], out[:, 1]

    def _to_physical(self, yx, yv, sx, sv):
        dx = soft_clamp_norm(yx * sx, self.cfg.max_disp_nm)
        dv = soft_clamp_norm(yv * sv, self.cfg.max_dvel_nm_per_ps)
        return dx, dv

    # -- training --------------------------------------------------------------
    def loss(self, batch: Batch, lambda_vel: float = 1.0, generator: Optional[torch.Generator] = None):
        z = batch["atom_types"]
        G = self._num_graphs(batch)
        _, sx, sv = self._scales(z)
        tx = (batch["x_tk"] - batch["x_t"]) / sx
        tv = (batch["v_tk"] - batch["v_t"]) / sv
        if self.cfg.kind == "mean":
            px, pv = self._field(batch, None, None, None)
            px, pv = self._project(px, pv, batch, sx, sv, G)
            lx = ((px - tx) ** 2).mean()
            lv = ((pv - tv) ** 2).mean()
        else:
            nx = torch.randn(tx.shape, device=tx.device, dtype=tx.dtype, generator=generator)
            nv = torch.randn(tv.shape, device=tv.device, dtype=tv.dtype, generator=generator)
            nx, nv = self._project(nx, nv, batch, sx, sv, G)
            tau_g = torch.rand(G, device=tx.device, dtype=tx.dtype, generator=generator)
            tau = tau_g[batch["batch"]]
            t_ = tau.unsqueeze(-1)
            yx = (1 - t_) * nx + t_ * tx
            yv = (1 - t_) * nv + t_ * tv
            ux, uv = self._field(batch, yx, yv, tau)
            ux, uv = self._project(ux, uv, batch, sx, sv, G)
            lx = ((ux - (tx - nx)) ** 2).mean()
            lv = ((uv - (tv - nv)) ** 2).mean()
        total = lx + lambda_vel * lv
        return total, {"loss_pos": float(lx.detach()), "loss_vel": float(lv.detach())}

    # -- inference -------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        batch: Batch,
        generator: Optional[torch.Generator] = None,
        steps: Optional[int] = None,
        solver: Optional[str] = None,
        noise_scale: float = 1.0,
        noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Δx, Δv) in physical units. Stochastic for ``kind == 'flow'``.

        ``noise`` optionally supplies the normalised starting noise (yx, yv) of the flow;
        ``edge_index`` a precomputed :func:`full_graph` (both needed for CUDA-graph capture).
        """
        z = batch["atom_types"]
        G = self._num_graphs(batch)
        _, sx, sv = self._scales(z)
        if edge_index is None:
            edge_index = full_graph(batch["batch"])
        if self.cfg.kind == "mean":
            px, pv = self._field(batch, None, None, None, edge_index)
            px, pv = self._project(px, pv, batch, sx, sv, G)
            return self._to_physical(px, pv, sx, sv)

        steps = int(steps or self.cfg.flow_steps)
        solver = (solver or self.cfg.flow_solver).lower()
        shape = batch["x_t"].shape
        if noise is None:
            yx = noise_scale * torch.randn(shape, device=sx.device, dtype=sx.dtype, generator=generator)
            yv = noise_scale * torch.randn(shape, device=sx.device, dtype=sx.dtype, generator=generator)
        else:
            yx, yv = noise
        yx, yv = self._project(yx, yv, batch, sx, sv, G)
        n = z.numel()

        def field(yx_, yv_, t: float):
            tau = torch.full((n,), t, device=sx.device, dtype=sx.dtype)
            ux, uv = self._field(batch, yx_, yv_, tau, edge_index)
            return self._project(ux, uv, batch, sx, sv, G)

        dt = 1.0 / steps
        for k in range(steps):
            t0 = k * dt
            ux, uv = field(yx, yv, t0)
            if solver == "heun":
                ex, ev = yx + dt * ux, yv + dt * uv
                ux2, uv2 = field(ex, ev, t0 + dt)
                yx = yx + 0.5 * dt * (ux + ux2)
                yv = yv + 0.5 * dt * (uv + uv2)
            else:
                yx, yv = yx + dt * ux, yv + dt * uv
        return self._to_physical(yx, yv, sx, sv)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT = "kstep-v2"


def build_model(cfg: Dict | ModelConfig) -> KStepModel:
    if not isinstance(cfg, ModelConfig):
        cfg = ModelConfig.from_dict(cfg)
    return KStepModel(cfg)


def save_checkpoint(path: Path, model: KStepModel, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT, "model_config": asdict(model.cfg), "model": model.state_dict()}
    payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_predictor(path: Path, device: torch.device | str = "cpu", legacy_model_config: Optional[Dict] = None,
                   use_ema: bool = True):
    """Load a checkpoint and return an object with ``predict(batch, generator=None)``.

    New (``kstep-v2``) checkpoints are self-describing. Older checkpoints from the
    Dec 2025 EGNN / Transformer-EGNN models need ``legacy_model_config`` and are
    wrapped in :class:`kstep.legacy_model.LegacyPredictor`.
    """
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
        model = build_model(payload["model_config"])
        state = payload.get("ema") if (use_ema and payload.get("ema")) else payload["model"]
        model.load_state_dict(state)
        model.to(device).eval()
        model.k_steps = payload.get("k_steps")
        model.checkpoint_meta = {k: v for k, v in payload.items() if k not in ("model", "ema", "optimizer", "scheduler")}
        return model
    if legacy_model_config is None:
        raise ValueError(f"{path} is a legacy checkpoint; pass the matching model YAML (--model-config)")
    from .legacy_model import LegacyPredictor

    return LegacyPredictor.from_checkpoint(payload, legacy_model_config, device)
