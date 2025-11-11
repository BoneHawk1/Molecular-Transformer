from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2, act=nn.SiLU) -> nn.Sequential:
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            net.append(act())
    return nn.Sequential(*net)


class SimpleET(nn.Module):
    """A tiny equivariant-ish model using pairwise messages.

    This is a lightweight stand-in for TorchMD-Net ET. It uses a
    learned radial function and message passing to produce per-atom
    vector outputs (deltas for x and v).
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff_nm: float = 0.7,
        tanh_head: bool = True,
        max_disp_nm: float = 0.02,
        use_force_head: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff_nm
        self.emb = nn.Embedding(num_atom_types, hidden_channels)
        self.message_mlp = make_mlp(hidden_channels + 1, hidden_channels, hidden_channels, layers=2)
        self.update_mlp = make_mlp(hidden_channels, hidden_channels, hidden_channels, layers=2)
        self.vec_head_dx = nn.Linear(hidden_channels, 3)
        self.vec_head_dv = nn.Linear(hidden_channels, 3)
        self.use_force_head = use_force_head
        self.force_head = nn.Linear(hidden_channels, 3) if use_force_head else None
        self.tanh_head = tanh_head
        self.max_disp = max_disp_nm

    def forward(
        self,
        x_nm: torch.Tensor,  # [B,N,3]
        atom_types: torch.Tensor,  # [B,N]
        mask: torch.Tensor,  # [B,N]
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = x_nm.shape
        h = self.emb(atom_types)  # [B,N,C]

        # Pairwise differences
        diff = x_nm[:, :, None, :] - x_nm[:, None, :, :]  # [B,N,N,3]
        dist = (diff.square().sum(-1) + 1e-9).sqrt()  # [B,N,N]

        # cutoff mask
        eye = torch.eye(N, dtype=torch.bool, device=x_nm.device)[None]
        valid = (dist < self.cutoff) & (~eye)
        if mask is not None:
            valid = valid & (mask[:, :, None] & mask[:, None, :])

        # Radial feature: 1/r with cutoff smooth mask
        radial = torch.where(valid, 1.0 / dist.clamp(min=1e-3), torch.zeros_like(dist))  # [B,N,N]
        radial = radial.unsqueeze(-1)  # [B,N,N,1]

        # Messages: concatenate sender features with radial
        sender_h = h[:, None, :, :].expand(B, N, N, h.size(-1))
        msg_in = torch.cat([sender_h, radial], dim=-1)
        msg = self.message_mlp(msg_in)  # [B,N,N,C]
        # Directional weighting
        direction = torch.where(valid.unsqueeze(-1), diff / dist.unsqueeze(-1).clamp(min=1e-6), torch.zeros_like(diff))
        vec_msg = (msg * direction).sum(dim=2)  # aggregate over neighbors -> [B,N,C]

        h = h + self.update_mlp(vec_msg)

        dx = self.vec_head_dx(h)  # [B,N,3]
        dv = self.vec_head_dv(h)

        if self.tanh_head:
            dx = torch.tanh(dx) * self.max_disp
            dv = torch.tanh(dv) * (self.max_disp / 0.02)  # scale similarly

        out = {"delta_x": dx, "delta_v": dv}
        if self.use_force_head and self.force_head is not None:
            out["force"] = self.force_head(h)
        return out


@dataclass
class ModelConfig:
    arch: str = "torchmdnet_et"
    hidden_channels: int = 128
    num_layers: int = 4
    num_heads: int = 4
    cutoff_nm: float = 0.7
    rbf_size: int = 16
    vec_channels: int = 16
    predict_delta: bool = True
    max_disp_nm: float = 0.02
    use_force_head: bool = False
    tanh_head: bool = True


def build_model(cfg: Dict, num_atom_types: int = 100) -> nn.Module:
    arch = cfg.get("arch", "torchmdnet_et")
    if arch == "torchmdnet_et":
        return SimpleET(
            num_atom_types=num_atom_types,
            hidden_channels=int(cfg.get("hidden_channels", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            cutoff_nm=float(cfg.get("cutoff_nm", 0.7)),
            tanh_head=bool(cfg.get("tanh_head", True)),
            max_disp_nm=float(cfg.get("max_disp_nm", 0.02)),
            use_force_head=bool(cfg.get("use_force_head", False)),
        )
    else:
        # Fallback to the simple model
        return SimpleET(
            num_atom_types=num_atom_types,
            hidden_channels=int(cfg.get("hidden_channels", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            cutoff_nm=float(cfg.get("cutoff_nm", 0.7)),
            tanh_head=bool(cfg.get("tanh_head", True)),
            max_disp_nm=float(cfg.get("max_disp_nm", 0.02)),
            use_force_head=bool(cfg.get("use_force_head", False)),
        )

