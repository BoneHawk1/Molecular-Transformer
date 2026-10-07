"""k-step window datasets.

On-disk format (``kstep-dataset-v2``, written by ``src/02_make_dataset.py``): all atoms
of all samples are concatenated into flat arrays and ``ptr`` holds sample offsets::

    x_t, v_t, x_tk, v_tk : (A, 3) float32   COM-centred, nm and nm/ps
    atom_types           : (A,)   int64
    masses               : (A,)   float32
    ptr                  : (S+1,) int64
    molecule             : (S,)   str
    start_frame          : (S,)   int64
    k_steps, frame_dt_ps : scalars

The legacy object-array format from 2025 is converted on load. Datasets this small
fit on the GPU, so batches are gathered there directly (no DataLoader workers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch

from .common import LOGGER, remove_com

DATASET_FORMAT = "kstep-dataset-v2"
ARRAY_KEYS = ("x_t", "v_t", "x_tk", "v_tk", "atom_types", "masses")


def build_windows(traj: Dict, k: int, stride: int, max_samples: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Cut a single trajectory into COM-centred (x_t, v_t) -> (x_{t+k}, v_{t+k}) windows."""
    pos, vel, masses = traj["pos"], traj["vel"], np.asarray(traj["masses"], dtype=np.float64)
    starts = np.arange(0, pos.shape[0] - k, stride)
    if max_samples > 0 and len(starts) > max_samples:
        starts = np.sort(rng.choice(starts, max_samples, replace=False))
    out = {key: [] for key in ("x_t", "v_t", "x_tk", "v_tk")}
    for s in starts:
        a = remove_com(pos[s].astype(np.float64), vel[s].astype(np.float64), masses)
        b = remove_com(pos[s + k].astype(np.float64), vel[s + k].astype(np.float64), masses)
        out["x_t"].append(a[0]); out["v_t"].append(a[1])
        out["x_tk"].append(b[0]); out["v_tk"].append(b[1])
    n_atoms = pos.shape[1]
    result = {key: np.concatenate(val).astype(np.float32) if val else np.zeros((0, 3), np.float32) for key, val in out.items()}
    result["atom_types"] = np.tile(np.asarray(traj["atom_types"], dtype=np.int64), len(starts))
    result["masses"] = np.tile(np.asarray(traj["masses"], dtype=np.float32), len(starts))
    result["n_atoms"] = np.full(len(starts), n_atoms, dtype=np.int64)
    result["start_frame"] = starts.astype(np.int64)
    return result


def save_dataset(path: Path, parts: Sequence[Dict[str, np.ndarray]], molecules: Sequence[str], k: int, frame_dt_ps: float) -> None:
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in ARRAY_KEYS}
    n_atoms = np.concatenate([p["n_atoms"] for p in parts])
    ptr = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int64)
    mol = np.concatenate([np.full(len(p["n_atoms"]), m) for p, m in zip(parts, molecules)])
    start = np.concatenate([p["start_frame"] for p in parts])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, ptr=ptr, molecule=mol, start_frame=start, k_steps=k,
             frame_dt_ps=frame_dt_ps, format=DATASET_FORMAT)
    LOGGER.info("Wrote %d samples (%d atoms) to %s", len(n_atoms), ptr[-1], path)


def _load_arrays(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "format" in data.files and str(data["format"]) == DATASET_FORMAT:
        out = {key: data[key] for key in data.files}
        out["k_steps"] = int(data["k_steps"])
        out["frame_dt_ps"] = float(data["frame_dt_ps"])
        return out
    # Legacy: one object array entry per sample
    LOGGER.info("Converting legacy object-array dataset %s", path)
    out = {key: np.concatenate([np.asarray(a) for a in data[key]]) for key in ARRAY_KEYS}
    out["atom_types"] = out["atom_types"].astype(np.int64)
    counts = np.array([len(a) for a in data["atom_types"]], dtype=np.int64)
    out["ptr"] = np.concatenate([[0], np.cumsum(counts)])
    out["molecule"] = data["molecule"]
    out["start_frame"] = np.full(len(counts), -1, dtype=np.int64)
    out["k_steps"] = int(data["k_steps"])
    out["frame_dt_ps"] = float("nan")
    return out


class KStepData:
    """In-memory dataset with vectorised batch gathering on any device."""

    def __init__(self, path: Path, molecules: Optional[Iterable[str]] = None, device: str | torch.device = "cpu") -> None:
        arrays = _load_arrays(Path(path))
        self.k_steps: int = arrays["k_steps"]
        self.frame_dt_ps: float = arrays["frame_dt_ps"]
        mol = np.asarray(arrays["molecule"]).astype(str)
        ptr = arrays["ptr"]
        keep = np.ones(len(mol), dtype=bool) if molecules is None else np.isin(mol, list(molecules))
        if not keep.any():
            raise ValueError(f"No samples in {path} for molecules {list(molecules or [])}")
        counts = np.diff(ptr)[keep]
        atom_idx = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in np.nonzero(keep)[0]])
        self.molecule = mol[keep]
        self.device = torch.device(device)
        self.tensors: Dict[str, torch.Tensor] = {}
        for key in ARRAY_KEYS:
            arr = arrays[key][atom_idx]
            dtype = torch.long if key == "atom_types" else torch.float32
            self.tensors[key] = torch.as_tensor(arr, dtype=dtype, device=self.device)
        self.counts = torch.as_tensor(counts, dtype=torch.long, device=self.device)
        self.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(self.counts, 0)])

    def __len__(self) -> int:
        return int(self.counts.numel())

    @property
    def molecules(self) -> List[str]:
        return sorted(set(self.molecule.tolist()))

    def gather(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build a batch dict for sample indices ``idx`` (on ``self.device``)."""
        idx = idx.to(self.device)
        counts = self.counts[idx]
        starts = self.ptr[idx]
        total = int(counts.sum())
        offsets = torch.cumsum(counts, 0) - counts
        within = torch.arange(total, device=self.device) - torch.repeat_interleave(offsets, counts)
        atoms = torch.repeat_interleave(starts, counts) + within
        batch = {key: t[atoms] for key, t in self.tensors.items()}
        batch["batch"] = torch.repeat_interleave(torch.arange(idx.numel(), device=self.device), counts)
        batch["num_graphs"] = idx.numel()
        return batch

    def iterate(self, batch_size: int, shuffle: bool = True, generator: Optional[torch.Generator] = None,
                limit: int = 0) -> Iterator[Dict[str, torch.Tensor]]:
        n = len(self)
        order = torch.randperm(n, generator=generator).to(self.device) if shuffle else torch.arange(n, device=self.device)
        for b, start in enumerate(range(0, n, batch_size)):
            if limit and b >= limit:
                break
            yield self.gather(order[start:start + batch_size])

    def normalization_tables(self, max_z: int, floor: float = 1e-4) -> Dict[str, torch.Tensor]:
        """Per-element RMS of v_t, Δx and Δv (global RMS for unseen elements)."""
        t = self.tensors
        z = t["atom_types"]
        quantities = {
            "vel_scale": t["v_t"],
            "dpos_scale": t["x_tk"] - t["x_t"],
            "dvel_scale": t["v_tk"] - t["v_t"],
        }
        tables = {}
        for name, q in quantities.items():
            sq = (q.double() ** 2).mean(-1)
            global_rms = float(sq.mean().sqrt())
            table = torch.full((max_z + 1,), max(global_rms, floor), dtype=torch.float64)
            sums = torch.zeros(max_z + 1, dtype=torch.float64, device=z.device).index_add_(0, z, sq)
            cnt = torch.zeros(max_z + 1, dtype=torch.float64, device=z.device).index_add_(0, z, torch.ones_like(sq))
            seen = cnt > 0
            table[seen.cpu()] = (sums[seen] / cnt[seen]).sqrt().clamp(min=floor).cpu()
            tables[name] = table.float()
        return tables
