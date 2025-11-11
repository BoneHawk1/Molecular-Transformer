"""Supervised training loop for the equivariant k-step model."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from utils import (configure_logging, ensure_dir, load_yaml, set_seed,
                   LOGGER)

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("md_kstep_model", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
import sys

sys.modules[_spec.name] = _module  # ensure dataclasses can resolve module
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]
build_model_from_config = _module.build_model_from_config  # type: ignore[attr-defined]

# Global cache for structural index tensors per molecule
STRUCT_INDEXES: Dict[str, Dict[str, np.ndarray]] = {}
# Torch-side cache of structural index tensors per (molecule, device)
STRUCT_IDX_TORCH: Dict[tuple[str, str], Dict[str, torch.Tensor]] = {}


@dataclass
class TrainConfig:
    seed: int
    k_steps: int
    batch_size: int
    grad_accum: int
    num_workers: int
    lr: float
    lr_min: float
    weight_decay: float
    max_epochs: int
    steps_per_epoch: int
    amp: bool
    grad_clip: float
    lambda_vel: float
    lambda_com: float
    lambda_force: float
    val_every_steps: int
    checkpoint_every_steps: int
    checkpoint_dir: str
    log_dir: str
    resume: str | None
    wandb: Dict
    random_rotate: bool
    # Rotation mode: 'batch' applies a single rotation to the whole batch,
    # 'per_graph' applies a different rotation per graph in the batch.
    random_rotate_mode: str = "batch"
    # Structural penalties (small weights recommended)
    lambda_struct_bond: float = 0.0
    lambda_struct_angle: float = 0.0
    lambda_struct_dihedral: float = 0.0
    # Optional caps for constraints per graph (None or <=0 to disable)
    struct_max_bonds: int | None = None
    struct_max_angles: int | None = None
    struct_max_dihedrals: int | None = None

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainConfig":
        return cls(**data)


class KStepDataset(Dataset):
    def __init__(self, dataset_path: Path, allowed_molecules: Iterable[str] | None = None):
        data = np.load(dataset_path, allow_pickle=True)
        self.x_t = data["x_t"]
        self.v_t = data["v_t"]
        self.x_tk = data["x_tk"]
        self.v_tk = data["v_tk"]
        self.masses = data["masses"]
        self.atom_types = data["atom_types"]
        self.molecule = data["molecule"]
        self.k_steps = int(data["k_steps"])

        if allowed_molecules is not None:
            mask = np.isin(self.molecule, list(allowed_molecules))
            self.x_t = self.x_t[mask]
            self.v_t = self.v_t[mask]
            self.x_tk = self.x_tk[mask]
            self.v_tk = self.v_tk[mask]
            self.masses = self.masses[mask]
            self.atom_types = self.atom_types[mask]
            self.molecule = self.molecule[mask]

        # Compute global std for deltas (for normalization in the loss)
        # Use scalar std across all components for stability
        pos_sq_sum = 0.0
        vel_sq_sum = 0.0
        count_pos = 0
        count_vel = 0
        for xt, xtk, vt, vtk in zip(self.x_t, self.x_tk, self.v_t, self.v_tk):
            dp = (xtk - xt).astype(np.float64)
            dv = (vtk - vt).astype(np.float64)
            pos_sq_sum += float(np.sum(dp * dp))
            vel_sq_sum += float(np.sum(dv * dv))
            count_pos += dp.size
            count_vel += dv.size
        # Avoid division by zero
        self.pos_std: float = float(np.sqrt(pos_sq_sum / max(count_pos, 1)))
        self.vel_std: float = float(np.sqrt(vel_sq_sum / max(count_vel, 1)))
        # Clamp to sensible minima to avoid over-amplifying the loss
        self.pos_std = float(max(self.pos_std, 1e-3))
        self.vel_std = float(max(self.vel_std, 1e-2))

        # Precompute structural index maps once per molecule to avoid per-step CPU work
        self.struct_index: Dict[str, Dict[str, np.ndarray]] = {}
        seen: set[str] = set()
        for idx in range(len(self.x_t)):
            mol = str(self.molecule[idx])
            if mol in seen:
                continue
            seen.add(mol)
            xt_ref = self.x_t[idx]
            types_ref = self.atom_types[idx]
            bonds = self._np_guess_bonds(xt_ref, types_ref)
            angles = self._np_build_angles(bonds)
            dihedrals = self._np_build_dihedrals(bonds)
            self.struct_index[mol] = {
                "bonds_i": np.array([i for i, _ in bonds], dtype=np.int64),
                "bonds_j": np.array([j for _, j in bonds], dtype=np.int64),
                "ang_i": np.array([i for i, _, _ in angles], dtype=np.int64),
                "ang_j": np.array([j for _, j, _ in angles], dtype=np.int64),
                "ang_k": np.array([k for _, _, k in angles], dtype=np.int64),
                "dih_a": np.array([a for a, _, _, _ in dihedrals], dtype=np.int64),
                "dih_b": np.array([b for _, b, _, _ in dihedrals], dtype=np.int64),
                "dih_c": np.array([c for _, _, c, _ in dihedrals], dtype=np.int64),
                "dih_d": np.array([d for _, _, _, d in dihedrals], dtype=np.int64),
            }

    def __len__(self) -> int:
        return len(self.x_t)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "x_t": self.x_t[idx],
            "v_t": self.v_t[idx],
            "x_tk": self.x_tk[idx],
            "v_tk": self.v_tk[idx],
            "masses": self.masses[idx],
            "atom_types": self.atom_types[idx],
            "molecule": str(self.molecule[idx]),
        }

    # ------- numpy-based index builders (run once per molecule) -------
    @staticmethod
    def _np_guess_bonds(x_nm: np.ndarray, atom_types: np.ndarray, tol_angstrom: float = 0.1) -> List[Tuple[int, int]]:
        x_A = x_nm * 10.0
        N = x_A.shape[0]
        radii = np.array([_COVALENT_RADII.get(int(t), 0.75) for t in atom_types], dtype=np.float32)
        ri = radii.reshape(-1, 1)
        rj = radii.reshape(1, -1)
        cutoff = ri + rj + tol_angstrom
        diff = x_A[:, None, :] - x_A[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        mask = (dist <= cutoff) & (~np.eye(N, dtype=bool))
        ii, jj = np.where(mask)
        bonds: List[Tuple[int, int]] = []
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i < j:
                bonds.append((i, j))
        return bonds

    @staticmethod
    def _np_build_angles(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        adj: Dict[int, List[int]] = {}
        for i, j in bonds:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        triples: List[Tuple[int, int, int]] = []
        for center, neigh in adj.items():
            L = len(neigh)
            if L < 2:
                continue
            for a in range(L):
                for b in range(a + 1, L):
                    triples.append((neigh[a], center, neigh[b]))
        return triples

    @staticmethod
    def _np_build_dihedrals(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        adj: Dict[int, List[int]] = {}
        for i, j in bonds:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        quads: List[Tuple[int, int, int, int]] = []
        for i, j in bonds:
            for k in adj.get(j, []):
                if k == i:
                    continue
                for l in adj.get(k, []):
                    if l == j or l == i:
                        continue
                    quads.append((i, j, k, l))
        seen = set()
        uniq: List[Tuple[int, int, int, int]] = []
        for a, b, c, d in quads:
            key = (a, b, c, d)
            rev = (d, c, b, a)
            if key in seen or rev in seen:
                continue
            seen.add(key)
            uniq.append(key)
        return uniq


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    x_t = [torch.from_numpy(item["x_t"]).to(device) for item in batch]
    v_t = [torch.from_numpy(item["v_t"]).to(device) for item in batch]
    x_tk = [torch.from_numpy(item["x_tk"]).to(device) for item in batch]
    v_tk = [torch.from_numpy(item["v_tk"]).to(device) for item in batch]
    masses = [torch.from_numpy(item["masses"]).to(device) for item in batch]
    atom_types = [torch.from_numpy(item["atom_types"]).long().to(device) for item in batch]
    molecules: List[str] = [item["molecule"] for item in batch]

    node_counts = [item.shape[0] for item in x_t]
    batch_index = torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(node_counts)])

    def _cat(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=0)

    return {
        "x_t": _cat(x_t),
        "v_t": _cat(v_t),
        "x_tk": _cat(x_tk),
        "v_tk": _cat(v_tk),
        "masses": _cat(masses),
        "atom_types": _cat(atom_types),
        "batch": batch_index,
        "node_slices": torch.tensor(node_counts, dtype=torch.long),
        "molecule": molecules,
    }


def _split_by_batch(tensor: torch.Tensor, node_slices: torch.Tensor) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    start = 0
    for count in node_slices.tolist():
        outputs.append(tensor[start:start + count])
        start += count
    return outputs


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if dim_size <= 0:
        return torch.zeros((0,) + src.shape[1:], device=src.device, dtype=src.dtype)
    out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def _com_loss(pred: torch.Tensor, target: torch.Tensor, masses: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    masses = masses.unsqueeze(-1)
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    total_mass = _scatter_sum(masses, batch_index, num_graphs)
    pred_com = _scatter_sum(pred * masses, batch_index, num_graphs) / torch.clamp(total_mass, min=1e-8)
    target_com = _scatter_sum(target * masses, batch_index, num_graphs) / torch.clamp(total_mass, min=1e-8)
    return torch.mean((pred_com - target_com) ** 2)


def _momentum_loss(pred_vel: torch.Tensor, target_vel: torch.Tensor, masses: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    masses = masses.unsqueeze(-1)
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    pred_momentum = _scatter_sum(pred_vel * masses, batch_index, num_graphs)
    target_momentum = _scatter_sum(target_vel * masses, batch_index, num_graphs)
    return torch.mean((pred_momentum - target_momentum) ** 2)


def _move_to(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val.to(device, non_blocking=True)
        else:
            out[key] = val
    return out


def _random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Shoemake (1992) uniform quaternions
    u1 = torch.rand((), device=device, dtype=dtype)
    u2 = torch.rand((), device=device, dtype=dtype)
    u3 = torch.rand((), device=device, dtype=dtype)
    sqrt_u1 = torch.sqrt(u1)
    sqrt_one_minus_u1 = torch.sqrt(1 - u1)
    theta1 = 2 * math.pi * u2
    theta2 = 2 * math.pi * u3
    x = sqrt_one_minus_u1 * torch.sin(theta1)
    y = sqrt_one_minus_u1 * torch.cos(theta1)
    z = sqrt_u1 * torch.sin(theta2)
    w = sqrt_u1 * torch.cos(theta2)
    rot = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])
    return rot.to(device=device, dtype=dtype)


def _apply_random_rotations(batch: Dict[str, torch.Tensor], mode: str = "batch") -> None:
    counts = batch["node_slices"].detach().cpu().tolist()
    if not counts:
        return
    device = batch["x_t"].device
    dtype = batch["x_t"].dtype
    if mode == "batch":
        rot = _random_rotation_matrix(device, dtype)
        for key in ("x_t", "v_t", "x_tk", "v_tk"):
            batch[key] = batch[key] @ rot.T
        return
    # Fallback: per-graph rotations (original behavior)
    start = 0
    for count in counts:
        end = start + count
        rot = _random_rotation_matrix(device, dtype)
        for key in ("x_t", "v_t", "x_tk", "v_tk"):
            batch[key][start:end] = batch[key][start:end] @ rot.T
        start = end


# --- Structural penalties utilities (torch) ---

# Covalent radii in Å (subset sufficient for this dataset)
_COVALENT_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}


def _guess_bonds_indices(x_nm: torch.Tensor, atom_types: torch.Tensor, tol_angstrom: float = 0.1) -> List[Tuple[int, int]]:
    # x_nm: (N,3) on the current device
    device = x_nm.device
    x_A = x_nm * 10.0
    N = x_A.shape[0]
    # Map atom_types to radii (fallback 0.75 Å)
    types_np = atom_types.detach().cpu().numpy().astype(int)
    radii_np = np.array([_COVALENT_RADII.get(t, 0.75) for t in types_np], dtype=np.float32)
    radii = torch.as_tensor(radii_np, device=device)
    # Pairwise thresholds
    ri = radii.view(-1, 1)
    rj = radii.view(1, -1)
    cutoff = ri + rj + tol_angstrom
    # Distances
    diff = x_A.unsqueeze(1) - x_A.unsqueeze(0)  # (N,N,3)
    dist = torch.linalg.norm(diff, dim=-1)  # (N,N)
    mask = (dist <= cutoff) & (~torch.eye(N, dtype=torch.bool, device=device))
    idx = torch.nonzero(mask, as_tuple=False)
    # Unique undirected pairs i<j
    pairs: List[Tuple[int, int]] = []
    for i, j in idx.tolist():
        if i < j:
            pairs.append((i, j))
    return pairs


def _build_angles_from_bonds(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    adj: Dict[int, List[int]] = {}
    for i, j in bonds:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    triples: List[Tuple[int, int, int]] = []
    for center, neigh in adj.items():
        L = len(neigh)
        if L < 2:
            continue
        for a in range(L):
            for b in range(a + 1, L):
                triples.append((neigh[a], center, neigh[b]))
    return triples


def _build_dihedrals_from_bonds(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    adj: Dict[int, List[int]] = {}
    for i, j in bonds:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    quads: List[Tuple[int, int, int, int]] = []
    for i, j in bonds:
        for k in adj.get(j, []):
            if k == i:
                continue
            for l in adj.get(k, []):
                if l == j or l == i:
                    continue
                quads.append((i, j, k, l))
    # Deduplicate by canonical key
    seen = set()
    uniq: List[Tuple[int, int, int, int]] = []
    for a, b, c, d in quads:
        key = (a, b, c, d)
        rev = (d, c, b, a)
        if key in seen or rev in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _get_struct_indices_torch(
    mol: str,
    device: torch.device,
    xt: torch.Tensor,
    types: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    key = (mol, str(device))
    cached = STRUCT_IDX_TORCH.get(key)
    if cached is not None:
        return cached
    out: Dict[str, torch.Tensor] = {}
    idxs_np = STRUCT_INDEXES.get(mol)
    if idxs_np is not None:
        # Convert precomputed numpy indices to torch on this device
        for k in ("bonds_i", "bonds_j", "ang_i", "ang_j", "ang_k", "dih_a", "dih_b", "dih_c", "dih_d"):
            arr = idxs_np.get(k)
            if arr is None or len(arr) == 0:
                out[k] = torch.tensor([], device=device, dtype=torch.long)
            else:
                out[k] = torch.as_tensor(arr, device=device, dtype=torch.long)
        STRUCT_IDX_TORCH[key] = out
        return out
    # Fallback: build from current positions/types once per (mol, device)
    bonds = _guess_bonds_indices(xt, types)
    angles = _build_angles_from_bonds(bonds) if bonds else []
    dihedrals = _build_dihedrals_from_bonds(bonds) if bonds else []
    out["bonds_i"] = (
        torch.as_tensor([i for i, _ in bonds], device=device, dtype=torch.long) if bonds else torch.tensor([], device=device, dtype=torch.long)
    )
    out["bonds_j"] = (
        torch.as_tensor([j for _, j in bonds], device=device, dtype=torch.long) if bonds else torch.tensor([], device=device, dtype=torch.long)
    )
    out["ang_i"] = (
        torch.as_tensor([i for i, _, _ in angles], device=device, dtype=torch.long) if angles else torch.tensor([], device=device, dtype=torch.long)
    )
    out["ang_j"] = (
        torch.as_tensor([j for _, j, _ in angles], device=device, dtype=torch.long) if angles else torch.tensor([], device=device, dtype=torch.long)
    )
    out["ang_k"] = (
        torch.as_tensor([k for _, _, k in angles], device=device, dtype=torch.long) if angles else torch.tensor([], device=device, dtype=torch.long)
    )
    out["dih_a"] = (
        torch.as_tensor([a for a, _, _, _ in dihedrals], device=device, dtype=torch.long) if dihedrals else torch.tensor([], device=device, dtype=torch.long)
    )
    out["dih_b"] = (
        torch.as_tensor([b for _, b, _, _ in dihedrals], device=device, dtype=torch.long) if dihedrals else torch.tensor([], device=device, dtype=torch.long)
    )
    out["dih_c"] = (
        torch.as_tensor([c for _, _, c, _ in dihedrals], device=device, dtype=torch.long) if dihedrals else torch.tensor([], device=device, dtype=torch.long)
    )
    out["dih_d"] = (
        torch.as_tensor([d for _, _, _, d in dihedrals], device=device, dtype=torch.long) if dihedrals else torch.tensor([], device=device, dtype=torch.long)
    )
    STRUCT_IDX_TORCH[key] = out
    return out


def _subsample_indices(
    device: torch.device,
    max_items: int | None,
    *tensors: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    if max_items is None or max_items <= 0 or len(tensors) == 0:
        return tensors, 1.0
    n = tensors[0].numel()
    if n == 0 or n <= max_items:
        return tensors, 1.0
    perm = torch.randperm(n, device=device)[:max_items]
    subs = tuple(t[perm] for t in tensors)
    scale = float(n) / float(max_items)
    return subs, scale

def _structural_losses(
    pred_pos: torch.Tensor,
    true_pos: torch.Tensor,
    atom_types: torch.Tensor,
    node_slices: torch.Tensor,
    mol_names: List[str],
    max_bonds: int | None = None,
    max_angles: int | None = None,
    max_dihedrals: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pred_pos.device
    bond_loss_sum = torch.zeros((), device=device)
    angle_loss_sum = torch.zeros((), device=device)
    dihedral_loss_sum = torch.zeros((), device=device)
    nb = 0
    na = 0
    nd = 0
    start = 0
    for idx_graph, count in enumerate(node_slices.tolist()):
        end = start + count
        xh = pred_pos[start:end]
        xt = true_pos[start:end]
        mol = mol_names[idx_graph]
        types = atom_types[start:end]
        idxs_t = _get_struct_indices_torch(mol, device, xt, types)
        bi = idxs_t["bonds_i"]
        bj = idxs_t["bonds_j"]
        ai = idxs_t["ang_i"]
        aj = idxs_t["ang_j"]
        ak = idxs_t["ang_k"]
        da = idxs_t["dih_a"]
        db = idxs_t["dih_b"]
        dc = idxs_t["dih_c"]
        dd = idxs_t["dih_d"]
        if bi.numel() > 0:
            # Optional subsampling of bonds per graph
            (i_idx, j_idx), scale_b = _subsample_indices(device, max_bonds, bi, bj)
            # Bond lengths in Å for a stable scale
            def bond_lengths(x):
                d = x[i_idx] - x[j_idx]
                return torch.linalg.norm(d, dim=-1) * 10.0
            bl_pred = bond_lengths(xh)
            bl_true = bond_lengths(xt)
            bond_loss_sum = bond_loss_sum + torch.mean((bl_pred - bl_true) ** 2) * scale_b
            nb += 1
        if ai.numel() > 0:
            # Optional subsampling of angles per graph
            (ii, jj, kk), scale_a = _subsample_indices(device, max_angles, ai, aj, ak)
            def angles_deg(x):
                v1 = x[ii] - x[jj]
                v2 = x[kk] - x[jj]
                v1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-9)
                v2 = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-9)
                cos = torch.clamp((v1 * v2).sum(dim=-1), -1.0, 1.0)
                return torch.acos(cos) * (180.0 / math.pi)
            ang_pred = angles_deg(xh)
            ang_true = angles_deg(xt)
            angle_loss_sum = angle_loss_sum + torch.mean((ang_pred - ang_true) ** 2) * scale_a
            na += 1
        if da.numel() > 0:
            # Optional subsampling of dihedrals per graph
            (ai2, aj2, ak2, al2), scale_d = _subsample_indices(device, max_dihedrals, da, db, dc, dd)
            def dihed_deg(x):
                b0 = x[aj2] - x[ai2]
                b1 = x[ak2] - x[aj2]
                b2 = x[al2] - x[ak2]
                b1n = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-9)
                v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
                w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
                xnum = (v * w).sum(dim=-1)
                yden = torch.linalg.norm(torch.cross(v, b1n, dim=-1), dim=-1) * (torch.linalg.norm(w, dim=-1) + 1e-9)
                ang = torch.atan2(yden, xnum + 1e-9) * (180.0 / math.pi)
                return ang
            dih_pred = dihed_deg(xh)
            dih_true = dihed_deg(xt)
            dihedral_loss_sum = dihedral_loss_sum + torch.mean((dih_pred - dih_true) ** 2) * scale_d
            nd += 1
        start = end
    bond_loss = bond_loss_sum / max(nb, 1)
    angle_loss = angle_loss_sum / max(na, 1)
    dihedral_loss = dihedral_loss_sum / max(nd, 1)
    return bond_loss, angle_loss, dihedral_loss


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    pos_std: float,
    vel_std: float,
) -> Dict[str, float]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_to(batch, device)
            outputs = model(batch)
            pred_pos = batch["x_t"] + outputs["delta_pos"]
            pred_vel = batch["v_t"] + outputs["delta_vel"]
            # Standardized delta losses
            loss_pos = torch.mean(((pred_pos - batch["x_t"]) / pos_std - (batch["x_tk"] - batch["x_t"]) / pos_std) ** 2)
            loss_vel = torch.mean(((pred_vel - batch["v_t"]) / vel_std - (batch["v_tk"] - batch["v_t"]) / vel_std) ** 2)
            loss = loss_pos + cfg.lambda_vel * loss_vel
            # Optional structural penalties (use small weights; computed on positions only)
            if cfg.lambda_struct_bond > 0 or cfg.lambda_struct_angle > 0 or cfg.lambda_struct_dihedral > 0:
                b_loss, a_loss, d_loss = _structural_losses(
                    pred_pos,
                    batch["x_tk"],
                    batch["atom_types"],
                    batch["node_slices"],
                    batch["molecule"],
                    getattr(cfg, "struct_max_bonds", None),
                    getattr(cfg, "struct_max_angles", None),
                    getattr(cfg, "struct_max_dihedrals", None),
                )
                loss = (
                    loss
                    + cfg.lambda_struct_bond * b_loss
                    + cfg.lambda_struct_angle * a_loss
                    + cfg.lambda_struct_dihedral * d_loss
                )
            losses.append(loss.item())
    model.train()
    return {"val_loss": float(np.mean(losses)) if losses else float("nan")}


def save_checkpoint(state: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset npz path (single k)")
    parser.add_argument("--model-config", type=Path, required=True, help="Model YAML config")
    parser.add_argument("--train-config", type=Path, required=True, help="Training YAML config")
    parser.add_argument("--splits", type=Path, nargs=2, required=True, help="Train and val split JSON files")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return parser


def load_split(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    return data.get("molecules", [])


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()

    model_cfg = load_yaml(args.model_config)
    train_cfg = TrainConfig.from_dict(load_yaml(args.train_config))
    set_seed(train_cfg.seed)

    # Redirect checkpoints for transformer architecture to a dedicated folder
    try:
        arch = str(model_cfg.get("arch", "egnn")).lower()
        if arch in ("transformer_egnn", "attention_egnn", "egnn_transformer"):
            ckpt_dir = Path(train_cfg.checkpoint_dir)
            name = ckpt_dir.name
            if name.startswith("checkpoints"):
                new_name = name.replace("checkpoints", "checkpoints_transformer", 1)
                ckpt_dir = ckpt_dir.with_name(new_name)
            else:
                ckpt_dir = ckpt_dir.parent / "checkpoints_transformer"
            train_cfg.checkpoint_dir = str(ckpt_dir)
    except Exception:
        pass

    log_dir = Path(train_cfg.log_dir)
    ensure_dir(log_dir)
    train_log_path = log_dir / "train_metrics.jsonl"
    val_log_path = log_dir / "val_metrics.jsonl"
    train_log_path.unlink(missing_ok=True)
    val_log_path.unlink(missing_ok=True)

    train_metrics: List[Dict[str, float]] = []
    val_metrics: List[Dict[str, float]] = []

    train_split = load_split(args.splits[0])
    val_split = load_split(args.splits[1])

    train_dataset = KStepDataset(args.dataset, train_split)
    val_dataset = KStepDataset(args.dataset, val_split)
    # Expose structural index cache globally for quick lookup
    global STRUCT_INDEXES
    STRUCT_INDEXES = train_dataset.struct_index
    pos_std = float(train_dataset.pos_std)
    vel_std = float(train_dataset.vel_std)

    device = torch.device(args.device)

    model = build_model_from_config(model_cfg)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )

    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = train_cfg.max_epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_cfg.lr_min)
    scaler = GradScaler("cuda", enabled=train_cfg.amp)

    global_step = 0
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(train_cfg.max_epochs):
        LOGGER.info("Epoch %d/%d", epoch + 1, train_cfg.max_epochs)
        for batch in train_loader:
            batch = _move_to(batch, device)
            if train_cfg.random_rotate:
                _apply_random_rotations(batch, getattr(train_cfg, "random_rotate_mode", "batch"))
            com_pos_loss = None
            momentum_loss = None
            force_reg = None
            with autocast("cuda", enabled=train_cfg.amp):
                outputs = model(batch)
                pred_pos = batch["x_t"] + outputs["delta_pos"]
                pred_vel = batch["v_t"] + outputs["delta_vel"]
                # Standardized delta losses (per-component scalar std)
                loss_pos = torch.mean(((pred_pos - batch["x_t"]) / pos_std - (batch["x_tk"] - batch["x_t"]) / pos_std) ** 2)
                loss_vel = torch.mean(((pred_vel - batch["v_t"]) / vel_std - (batch["v_tk"] - batch["v_t"]) / vel_std) ** 2)
                loss = loss_pos + train_cfg.lambda_vel * loss_vel
                # Structural penalties
                if (
                    train_cfg.lambda_struct_bond > 0
                    or train_cfg.lambda_struct_angle > 0
                    or train_cfg.lambda_struct_dihedral > 0
                ):
                    b_loss, a_loss, d_loss = _structural_losses(
                        pred_pos,
                        batch["x_tk"],
                        batch["atom_types"],
                        batch["node_slices"],
                        batch["molecule"],
                        getattr(train_cfg, "struct_max_bonds", None),
                        getattr(train_cfg, "struct_max_angles", None),
                        getattr(train_cfg, "struct_max_dihedrals", None),
                    )
                    loss = (
                        loss
                        + train_cfg.lambda_struct_bond * b_loss
                        + train_cfg.lambda_struct_angle * a_loss
                        + train_cfg.lambda_struct_dihedral * d_loss
                    )
                if train_cfg.lambda_com > 0:
                    com_pos_loss = _com_loss(pred_pos, batch["x_tk"], batch["masses"], batch["batch"])
                    momentum_loss = _momentum_loss(pred_vel, batch["v_tk"], batch["masses"], batch["batch"])
                    loss = loss + train_cfg.lambda_com * (com_pos_loss + momentum_loss)
                if train_cfg.lambda_force > 0 and "force_pred" in outputs:
                    force_reg = torch.mean(outputs["force_pred"] ** 2)
                    loss = loss + train_cfg.lambda_force * force_reg

            current_step = global_step + 1
            metrics_entry: Dict[str, float | int] = {
                "step": int(current_step),
                "epoch": int(epoch + 1),
                "loss": float(loss.detach().item()),
                "loss_pos": float(loss_pos.detach().item()),
                "loss_vel": float(loss_vel.detach().item()),
            }
            if com_pos_loss is not None and momentum_loss is not None:
                metrics_entry["loss_com"] = float(com_pos_loss.detach().item())
                metrics_entry["loss_momentum"] = float(momentum_loss.detach().item())
            if force_reg is not None:
                metrics_entry["loss_force"] = float(force_reg.detach().item())
            train_metrics.append(metrics_entry)

            scaler.scale(loss / train_cfg.grad_accum).backward()

            if (global_step + 1) % train_cfg.grad_accum == 0:
                if train_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if train_cfg.val_every_steps > 0 and global_step % train_cfg.val_every_steps == 0:
                metrics = run_validation(model, val_loader, device, train_cfg, pos_std, vel_std)
                LOGGER.info("Step %d: val_loss=%.6f", global_step, metrics["val_loss"])
                val_metrics.append({
                    "step": int(global_step),
                    "epoch": int(epoch + 1),
                    "val_loss": float(metrics["val_loss"]),
                })
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    checkpoint_path = Path(train_cfg.checkpoint_dir) / "best.pt"
                    save_checkpoint({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": best_val,
                    }, checkpoint_path)

            if train_cfg.checkpoint_every_steps > 0 and global_step % train_cfg.checkpoint_every_steps == 0:
                checkpoint_path = Path(train_cfg.checkpoint_dir) / f"step_{global_step}.pt"
                save_checkpoint({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                }, checkpoint_path)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    _write_jsonl(train_log_path, train_metrics)
    _write_jsonl(val_log_path, val_metrics)

    LOGGER.info("Training complete. Best val_loss=%.6f", best_val)


if __name__ == "__main__":
    main()
