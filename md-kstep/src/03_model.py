"""Equivariant neural network predicting Δx/Δv for k-step MD jumps."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
from torch import nn

try:
    from torch_geometric.nn import radius_graph as pyg_radius_graph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyg_radius_graph = None

LOGGER = logging.getLogger("md_kstep.model")


@dataclass
class ModelConfig:
    arch: str = "egnn"
    hidden_dim: int = 128
    num_layers: int = 4
    cutoff_nm: float = 0.7
    activation: str = "silu"
    predict_delta: bool = True
    max_disp_nm: float = 0.02
    max_dvel_nm_per_ps: float = 0.1
    use_force_head: bool = False
    force_head_weight: float = 0.1
    dropout: float = 0.0
    embedding_dim: int = 64
    layer_norm: bool = True
    # Transformer-EGNN specific (kept optional for backward-compat)
    attention_heads: int = 8
    use_edge_attention: bool = True
    attention_dropout: float = 0.0
    positional_encoding: str = "none"  # "none" | "learned"
    feedforward_dim: Optional[int] = None  # defaults to 4*hidden_dim if None/0
    attention_type: str = "multi_head"  # placeholder for future variants
    use_cross_attention: bool = False   # reserved flag; not required for single-stream updates
    # QM-specific options
    use_qm_features: bool = False  # Enable QM electronic structure features
    electronic_dim: int = 64  # Dimension for electronic structure encoding
    num_orbitals: int = 0  # Number of orbitals (0 = auto-detect from input)
    predict_electronic: bool = False  # Predict electronic structure updates
    lambda_electronic: float = 0.1  # Weight for electronic structure loss
    enforce_orthogonality: bool = False  # Enforce orbital orthogonality constraint


def _activation(name: str) -> nn.Module:
    if name.lower() == "silu":
        return nn.SiLU()
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, activation: nn.Module, layer_norm: bool, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

    @staticmethod
    def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        counts = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
        counts.index_add_(0, index, torch.ones((src.size(0), 1), device=src.device, dtype=src.dtype))
        counts = torch.clamp(counts, min=1.0)
        return out / counts

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        rel = x[row] - x[col]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
        edge_input = torch.cat([h[row], h[col], dist2], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        coord_gate = self.coord_mlp(edge_feat)
        coord_update = rel * coord_gate
        delta_x = self._scatter_mean(coord_update, row, x.size(0))
        x = x + delta_x

        agg = self._scatter_mean(edge_feat, row, h.size(0))
        node_input = torch.cat([h, agg], dim=-1)
        delta_h = self.node_mlp(node_input)
        delta_h = self.dropout(delta_h)
        h = self.norm(h + delta_h)
        h = self.activation(h)
        return x, h


class EquivariantKStepModel(nn.Module):
    def __init__(self, cfg: ModelConfig, num_atom_types: int = 100) -> None:
        super().__init__()
        self.cfg = cfg
        self.atom_emb = nn.Embedding(num_atom_types, cfg.embedding_dim)
        self.vel_proj = nn.Linear(3, cfg.embedding_dim)
        self.mass_proj = nn.Linear(1, cfg.embedding_dim)

        hidden = cfg.hidden_dim
        self.node_proj = nn.Linear(cfg.embedding_dim * 3, hidden)
        activation = _activation(cfg.activation)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden, activation, cfg.layer_norm, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        self.head_pos = nn.Linear(hidden, 3)
        self.head_vel = nn.Linear(hidden, 3)
        if cfg.use_force_head:
            self.head_force = nn.Linear(hidden, 3)
        else:
            self.head_force = None

    @staticmethod
    def _dense_radius_graph(pos: torch.Tensor, batch: torch.Tensor, cutoff: float) -> torch.Tensor:
        device = pos.device
        batch = batch.to(device)
        edge_list = []
        unique_batches = batch.unique(sorted=True)
        cutoff_sq = cutoff * cutoff
        for b in unique_batches.tolist():
            mask = batch == b
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            coords = pos[idx]
            if coords.numel() == 0:
                continue
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1)
            adjacency = (dist2 <= cutoff_sq) & (~torch.eye(len(idx), dtype=torch.bool, device=device))
            edges = torch.nonzero(adjacency, as_tuple=False)
            if edges.numel() > 0:
                edge_list.append(torch.stack([idx[edges[:, 0]], idx[edges[:, 1]]], dim=0))
        if edge_list:
            return torch.cat(edge_list, dim=1)
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    def _build_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if pyg_radius_graph is not None:
            try:
                return pyg_radius_graph(pos, r=self.cfg.cutoff_nm, batch=batch, loop=False, max_num_neighbors=32)
            except Exception:
                pass  # fall back to dense implementation if torch-cluster is unavailable
        return self._dense_radius_graph(pos, batch, self.cfg.cutoff_nm)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos = batch["x_t"]
        vel = batch["v_t"]
        atom_types = batch["atom_types"]
        masses = batch["masses"].unsqueeze(-1)
        batch_index = batch["batch"]

        h = torch.cat([
            self.atom_emb(atom_types),
            self.vel_proj(vel),
            self.mass_proj(masses),
        ], dim=-1)
        h = self.node_proj(h)

        edge_index = self._build_graph(pos, batch_index)

        for layer in self.layers:
            pos, h = layer(pos, h, edge_index)

        delta_pos = self.head_pos(h)
        delta_vel = self.head_vel(h)

        if self.cfg.max_disp_nm > 0:
            delta_pos = torch.tanh(delta_pos) * self.cfg.max_disp_nm
        if self.cfg.max_dvel_nm_per_ps > 0:
            delta_vel = torch.tanh(delta_vel) * self.cfg.max_dvel_nm_per_ps

        outputs = {
            "delta_pos": delta_pos,
            "delta_vel": delta_vel,
        }
        if self.head_force is not None:
            outputs["force_pred"] = self.head_force(h)
        return outputs


class TransformerEGNNLayer(nn.Module):
    """
    Attention-weighted message passing with coordinate updates gated by edge features.
    Implements a lightweight transformer-like block on the neighborhood graph.
    """
    def __init__(
        self,
        hidden_dim: int,
        activation: nn.Module,
        layer_norm: bool,
        dropout: float,
        use_edge_attention: bool,
        attention_dropout: float,
        positional_encoding: str,
        cutoff_nm: float,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_edge_attention = use_edge_attention
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.positional_encoding = positional_encoding
        self.cutoff_nm = cutoff_nm
        self.num_rbfs = 16 if positional_encoding == "learned" else 0
        rbf_in = self.num_rbfs if self.num_rbfs > 0 else 1

        # Edge scoring (attention weight in [0,1])
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + rbf_in, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
        )
        # Coordinate gating
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + rbf_in, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
        )
        # Node update (message -> residual)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Feed-forward (transformer style)
        ff_dim = max(feedforward_dim, 4 * hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        if self.num_rbfs <= 0:
            return dist.unsqueeze(-1)  # fallback single scalar
        device = dist.device
        centers = torch.linspace(0.0, float(self.cutoff_nm), self.num_rbfs, device=device).view(1, -1)
        gamma = 10.0 / (self.cutoff_nm + 1e-6)
        d = dist.view(-1, 1) - centers
        return torch.exp(-gamma * d * d)

    @staticmethod
    def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        counts = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
        counts.index_add_(0, index, torch.ones((src.size(0), 1), device=src.device, dtype=src.dtype))
        counts = torch.clamp(counts, min=1.0)
        return out / counts

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index  # messages j->i with (i=row, j=col)
        rel = x[row] - x[col]
        dist2 = (rel ** 2).sum(dim=-1)
        dist = torch.sqrt(torch.clamp(dist2, min=0.0) + 1e-12)
        r_feat = self._rbf(dist)  # (E, R)

        # Build edge features for attention and coordinate gates
        edge_feat = torch.cat([h[row], h[col], r_feat], dim=-1)
        att_logits = self.att_mlp(edge_feat).squeeze(-1)  # (E,)
        if not self.use_edge_attention:
            att_logits = torch.zeros_like(att_logits)
        weights = torch.sigmoid(att_logits).unsqueeze(-1)  # (E,1)
        weights = self.attn_dropout(weights)

        # Aggregate value features and coordinate updates
        v = h[col]
        agg_val = self._scatter_mean(weights * v, row, h.size(0))

        coord_gate = torch.sigmoid(self.coord_mlp(edge_feat)).unsqueeze(-1)  # (E,1,1) after unsqueeze
        # For coordinates, a mean over weighted relative vectors
        delta_x = self._scatter_mean((weights * coord_gate.squeeze(-1)) * rel, row, x.size(0))
        x = x + delta_x

        # Residual node update
        node_input = torch.cat([h, agg_val], dim=-1)
        delta_h = self.msg_mlp(node_input)
        delta_h = self.dropout(delta_h)
        h = self.norm1(h + delta_h)
        h = self.activation(h)

        # Feed-forward sub-layer
        h = self.norm2(h + self.ffn(h))
        return x, h


class TransformerEGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, num_atom_types: int = 100) -> None:
        super().__init__()
        self.cfg = cfg
        self.atom_emb = nn.Embedding(num_atom_types, cfg.embedding_dim)
        self.vel_proj = nn.Linear(3, cfg.embedding_dim)
        self.mass_proj = nn.Linear(1, cfg.embedding_dim)

        hidden = cfg.hidden_dim
        self.node_proj = nn.Linear(cfg.embedding_dim * 3, hidden)
        activation = _activation(cfg.activation)
        feedforward_dim = (cfg.feedforward_dim or (4 * hidden))
        self.layers = nn.ModuleList([
            TransformerEGNNLayer(
                hidden_dim=hidden,
                activation=activation,
                layer_norm=cfg.layer_norm,
                dropout=cfg.dropout,
                use_edge_attention=cfg.use_edge_attention,
                attention_dropout=cfg.attention_dropout,
                positional_encoding=cfg.positional_encoding,
                cutoff_nm=cfg.cutoff_nm,
                feedforward_dim=feedforward_dim,
            )
            for _ in range(cfg.num_layers)
        ])
        self.head_pos = nn.Linear(hidden, 3)
        self.head_vel = nn.Linear(hidden, 3)
        if cfg.use_force_head:
            self.head_force = nn.Linear(hidden, 3)
        else:
            self.head_force = None

    @staticmethod
    def _dense_radius_graph(pos: torch.Tensor, batch: torch.Tensor, cutoff: float) -> torch.Tensor:
        device = pos.device
        batch = batch.to(device)
        edge_list = []
        unique_batches = batch.unique(sorted=True)
        cutoff_sq = cutoff * cutoff
        for b in unique_batches.tolist():
            mask = batch == b
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            coords = pos[idx]
            if coords.numel() == 0:
                continue
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1)
            adjacency = (dist2 <= cutoff_sq) & (~torch.eye(len(idx), dtype=torch.bool, device=device))
            edges = torch.nonzero(adjacency, as_tuple=False)
            if edges.numel() > 0:
                edge_list.append(torch.stack([idx[edges[:, 0]], idx[edges[:, 1]]], dim=0))
        if edge_list:
            return torch.cat(edge_list, dim=1)
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    def _build_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if pyg_radius_graph is not None:
            try:
                return pyg_radius_graph(pos, r=self.cfg.cutoff_nm, batch=batch, loop=False, max_num_neighbors=32)
            except Exception:
                pass
        return self._dense_radius_graph(pos, batch, self.cfg.cutoff_nm)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos = batch["x_t"]
        vel = batch["v_t"]
        atom_types = batch["atom_types"]
        masses = batch["masses"].unsqueeze(-1)
        batch_index = batch["batch"]

        h = torch.cat([
            self.atom_emb(atom_types),
            self.vel_proj(vel),
            self.mass_proj(masses),
        ], dim=-1)
        h = self.node_proj(h)

        edge_index = self._build_graph(pos, batch_index)
        for layer in self.layers:
            pos, h = layer(pos, h, edge_index)

        delta_pos = self.head_pos(h)
        delta_vel = self.head_vel(h)

        if self.cfg.max_disp_nm > 0:
            delta_pos = torch.tanh(delta_pos) * self.cfg.max_disp_nm
        if self.cfg.max_dvel_nm_per_ps > 0:
            delta_vel = torch.tanh(delta_vel) * self.cfg.max_dvel_nm_per_ps

        outputs = {
            "delta_pos": delta_pos,
            "delta_vel": delta_vel,
        }
        if self.head_force is not None:
            outputs["force_pred"] = self.head_force(h)
        return outputs


class ElectronicStructureEncoder(nn.Module):
    """Encoder for electronic structure variables (orbitals, density matrix)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: nn.Module):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, electronic_input: torch.Tensor) -> torch.Tensor:
        """Encode electronic structure input to hidden representation."""
        return self.encoder(electronic_input)


class QMCrossAttention(nn.Module):
    """Cross-attention between nuclear and electronic features."""
    def __init__(self, nuclear_dim: int, electronic_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(nuclear_dim, hidden_dim)
        self.k_proj = nn.Linear(electronic_dim, hidden_dim)
        self.v_proj = nn.Linear(electronic_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, nuclear_dim)
        
    def forward(self, nuclear_features: torch.Tensor, electronic_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nuclear_features: [B, N, nuclear_dim] or [N, nuclear_dim]
            electronic_features: [B, M, electronic_dim] or [M, electronic_dim] where M is number of orbitals/electronic DOF
        Returns:
            Enhanced nuclear features: [B, N, nuclear_dim] or [N, nuclear_dim]
        """
        if nuclear_features.dim() == 2:
            nuclear_features = nuclear_features.unsqueeze(0)
            electronic_features = electronic_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = nuclear_features.shape
        _, M, _ = electronic_features.shape
        
        q = self.q_proj(nuclear_features).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(electronic_features).view(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(electronic_features).view(B, M, self.num_heads, self.head_dim)
        
        # Attention: Q @ K^T / sqrt(d)
        scores = torch.einsum('bnhd,bmhd->bnmh', q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=2)
        
        # Apply attention to values
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v)
        out = out.contiguous().view(B, N, -1)
        out = self.out_proj(out)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class QMEnhancedModel(nn.Module):
    """EGNN model enhanced with QM electronic structure features."""
    def __init__(self, cfg: ModelConfig, num_atom_types: int = 100, num_orbitals: int = 0):
        super().__init__()
        self.cfg = cfg
        self.base_model = EquivariantKStepModel(cfg, num_atom_types)
        
        if cfg.use_qm_features:
            self.electronic_encoder = ElectronicStructureEncoder(
                input_dim=num_orbitals if num_orbitals > 0 else 64,
                hidden_dim=cfg.electronic_dim,
                output_dim=cfg.electronic_dim,
                activation=_activation(cfg.activation),
            )
            self.qm_attention = QMCrossAttention(
                nuclear_dim=cfg.hidden_dim,
                electronic_dim=cfg.electronic_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.attention_heads // 2,
            )
            
            if cfg.predict_electronic:
                self.head_electronic = nn.Linear(cfg.hidden_dim, num_orbitals if num_orbitals > 0 else 64)
        else:
            self.electronic_encoder = None
            self.qm_attention = None
            self.head_electronic = None
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with optional QM features."""
        outputs = self.base_model(batch)
        
        if self.cfg.use_qm_features and "electronic_t" in batch:
            # Encode electronic structure
            electronic_encoded = self.electronic_encoder(batch["electronic_t"])
            
            # Apply cross-attention
            nuclear_features = self.base_model.node_proj(
                torch.cat([
                    self.base_model.atom_emb(batch["atom_types"]),
                    self.base_model.vel_proj(batch["v_t"]),
                    self.base_model.mass_proj(batch["masses"].unsqueeze(-1)),
                ], dim=-1)
            )
            enhanced_nuclear = self.qm_attention(nuclear_features, electronic_encoded)
            
            # Predict electronic updates if requested
            if self.cfg.predict_electronic and self.head_electronic is not None:
                outputs["delta_electronic"] = self.head_electronic(enhanced_nuclear.mean(dim=1))
        
        return outputs


def build_model_from_config(cfg_dict: Dict) -> nn.Module:
    cfg = ModelConfig(**cfg_dict)
    arch = (cfg.arch or "egnn").lower()
    num_orbitals = cfg_dict.get("num_orbitals", 0)
    
    if cfg.use_qm_features:
        if arch in ("egnn", "basic_egnn"):
            return QMEnhancedModel(cfg, num_atom_types=100, num_orbitals=num_orbitals)
        # For transformer, we'd need a QM-enhanced version too
        LOGGER.warning("QM features not yet implemented for transformer_egnn, using base model")
    
    if arch in ("egnn", "basic_egnn"):
        return EquivariantKStepModel(cfg)
    if arch in ("transformer_egnn", "attention_egnn", "egnn_transformer"):
        return TransformerEGNN(cfg)
    raise ValueError(f"Unsupported architecture '{cfg.arch}'. Supported: 'egnn', 'transformer_egnn'.")
