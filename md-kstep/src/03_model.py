"""Equivariant neural network predicting Δx/Δv for k-step MD jumps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

try:
    from torch_geometric.nn import radius_graph as pyg_radius_graph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyg_radius_graph = None


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


def build_model_from_config(cfg_dict: Dict) -> EquivariantKStepModel:
    cfg = ModelConfig(**cfg_dict)
    return EquivariantKStepModel(cfg)
