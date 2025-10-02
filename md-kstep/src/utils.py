"""Utility helpers for md-kstep project.

This module centralizes:
- Seeding and device helpers
- YAML and JSON I/O
- Simple logging setup
- Units and center-of-mass utilities
- Neighbor graph and RDF computation
- Dataloader collation with padding and masks

All functions are written to be framework-agnostic except where noted.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing at docs time
    torch = None  # type: ignore


# -------------------------
# General I/O and logging
# -------------------------


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: os.PathLike | str, data: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: os.PathLike | str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: os.PathLike | str, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_npz(path: os.PathLike | str, **arrays: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def get_logger(name: str = "md-kstep"):
    import logging

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@contextmanager
def time_block(description: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    get_logger().info(f"{description} took {elapsed:.2f}s")


# -------------------------
# Reproducibility & device
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def get_device(prefer_gpu: bool = True) -> str:
    if torch is None:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"  # Apple Silicon
    return "cpu"


# -------------------------
# Units and COM utilities
# -------------------------


def center_of_mass(positions_nm: np.ndarray, masses: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute center of mass for each frame.

    positions_nm: [T, N, 3]
    masses: [N]
    mask: [N] or [T, N] boolean; if provided, excludes masked atoms
    Returns: [T, 3]
    """
    pos = np.asarray(positions_nm)
    m = np.asarray(masses)
    if mask is None:
        m_exp = m[None, :, None]
        total_mass = np.sum(m)
        return (pos * m_exp).sum(axis=1) / total_mass
    else:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.ndim == 1:
            mask_arr = np.broadcast_to(mask_arr[None, :], pos.shape[:2])
        m_masked = m[None, :] * mask_arr
        total_mass = m_masked.sum(axis=1)[:, None]
        m_exp = m_masked[:, :, None]
        # avoid divide by zero
        total_mass = np.where(total_mass == 0.0, 1.0, total_mass)
        return (pos * m_exp).sum(axis=1) / total_mass


def remove_com_translation(
    positions_nm: np.ndarray, masses: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    com = center_of_mass(positions_nm, masses, mask)  # [T,3]
    return positions_nm - com[:, None, :]


def remove_com_velocity(vel_nm_per_ps: np.ndarray, masses: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    # COM velocity v_com = sum(m_i v_i) / sum(m_i)
    vel = np.asarray(vel_nm_per_ps)
    m = np.asarray(masses)
    if mask is None:
        m_exp = m[None, :, None]
        total_mass = np.sum(m)
        vcom = (vel * m_exp).sum(axis=1) / total_mass
        return vel - vcom[:, None, :]
    else:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.ndim == 1:
            mask_arr = np.broadcast_to(mask_arr[None, :], vel.shape[:2])
        m_masked = m[None, :] * mask_arr
        total_mass = m_masked.sum(axis=1)[:, None]
        m_exp = m_masked[:, :, None]
        total_mass = np.where(total_mass == 0.0, 1.0, total_mass)
        vcom = (vel * m_exp).sum(axis=1) / total_mass
        return vel - vcom[:, None, :]


# -------------------------
# Neighbor graph and RDF
# -------------------------


def build_radius_graph(
    positions_nm: torch.Tensor,
    mask: torch.Tensor,
    cutoff_nm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build undirected radius graph edges.

    positions_nm: [B, N, 3]
    mask: [B, N] bool
    Returns: (row, col) each [E] with row->col edges (no self-edges). Includes both (i,j) and (j,i).
    """
    assert positions_nm.dim() == 3 and mask.dim() == 2
    B, N, _ = positions_nm.shape
    device = positions_nm.device
    cutoff = torch.tensor(cutoff_nm, device=device)
    # pairwise distances per batch
    # Compute squared distances with broadcasting; mask out invalid atoms
    pos = positions_nm  # [B,N,3]
    diff = pos[:, :, None, :] - pos[:, None, :, :]  # [B,N,N,3]
    dist2 = (diff * diff).sum(-1)  # [B,N,N]
    # mask
    valid = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(N, dtype=torch.bool, device=device)[None, :, :]
    valid = valid & (~eye)
    within = (dist2 <= cutoff * cutoff) & valid
    # collect edges
    batches, rows, cols = within.nonzero(as_tuple=True)  # [E]
    # convert (b,i,j) -> linear indexing per graph by offsetting rows/cols with batch index
    # We'll return (row_idx, col_idx) in flattened [B*N]
    row = batches * N + rows
    col = batches * N + cols
    return row, col


def compute_rdf(
    positions_nm: np.ndarray,
    mask: Optional[np.ndarray],
    r_max_nm: float,
    bin_width_nm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r) for a single trajectory.

    positions_nm: [T, N, 3]
    mask: [N] or [T, N] or None
    Returns: (r_centers, g_r)
    """
    pos = np.asarray(positions_nm)
    T, N, _ = pos.shape
    if mask is None:
        mask_arr = np.ones((T, N), dtype=bool)
    else:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.ndim == 1:
            mask_arr = np.broadcast_to(mask_arr[None, :], (T, N))

    # Pair distances per frame
    def frame_rdf(frame_pos: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
        idx = np.where(frame_mask)[0]
        if idx.size < 2:
            return np.array([])
        fp = frame_pos[idx]
        diff = fp[:, None, :] - fp[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        d = d[np.triu_indices(fp.shape[0], k=1)]  # upper triangle
        return d

    dists = [frame_rdf(pos[t], mask_arr[t]) for t in range(T)]
    dists = np.concatenate([d for d in dists if d.size > 0], axis=0) if dists else np.array([])

    bins = np.arange(0.0, r_max_nm + bin_width_nm, bin_width_nm)
    hist, edges = np.histogram(dists, bins=bins, density=False)
    # Normalize: ideal gas scaling ~ 4*pi*r^2 dr
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell_vol = 4.0 * math.pi * (centers ** 2) * bin_width_nm
    number_density = (np.sum(mask_arr) / T) / (4.0 / 3.0 * math.pi * (r_max_nm ** 3) + 1e-12)
    g_r = hist / (shell_vol * number_density * T + 1e-12)
    return centers, g_r


# -------------------------
# Dataloader collation
# -------------------------


def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch of variable-size molecular samples to a common N.

    Expects each sample to contain arrays/tensors with leading dimension N.
    Keys handled specially: 'mask' created from 'num_atoms' if not provided.
    """
    def to_tensor(x):
        if torch is None:
            return x
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)

    max_n = max(int(s.get("num_atoms", s["x_t"].shape[0])) for s in batch)

    def pad_array(arr: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
        # arr shape [N, ...]
        N = arr.shape[0]
        if N == max_n:
            return arr
        pad_width = [(0, max_n - N)] + [(0, 0) for _ in range(arr.ndim - 1)]
        return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=pad_value)

    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        if key in ("num_atoms", "molecule_id"):
            out[key] = np.array([s[key] for s in batch])
            continue
        values = [s[key] for s in batch]
        if isinstance(values[0], np.ndarray) and values[0].ndim >= 1:
            padded = [pad_array(v) for v in values]
            stacked = np.stack(padded, axis=0)
            out[key] = to_tensor(stacked)
        else:
            out[key] = to_tensor(np.array(values))

    # mask
    if "mask" not in out:
        mask_list = []
        for s in batch:
            n = int(s.get("num_atoms", s["x_t"].shape[0]))
            m = np.zeros((max_n,), dtype=bool)
            m[:n] = True
            mask_list.append(m)
        out["mask"] = to_tensor(np.stack(mask_list, axis=0))

    return out


@dataclass
class RolloutMetrics:
    rmsd_nm: float
    rmsv_nm_per_ps: float
    mean_delta_pos_nm: float
    mean_delta_vel_nm_per_ps: float


def compute_rms(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    diff = a - b
    if mask is not None:
        diff = diff[:, mask, :]
    return float(np.sqrt(np.mean(diff ** 2)))

