"""Utility helpers shared across md-kstep scripts."""
from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in some scripts
    yaml = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for some scripts
    torch = None  # type: ignore

LOGGER = logging.getLogger("md_kstep")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a terse logging format for CLI scripts."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def set_seed(seed: int) -> None:
    """Set python, numpy, and torch seeds (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict:
    """Load a YAML file into a dictionary."""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML files. Install it via `pip install pyyaml`.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(data: Dict, path: Path) -> None:
    """Write a dictionary to JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_smiles(path: Path) -> List[str]:
    """Read a SMILES file, skipping blank/commented lines."""
    smiles: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line)
    return smiles


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def numpy_seed(seed: Optional[int]) -> Iterator[None]:
    """Temporarily set the numpy RNG seed."""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute the center of mass for a single frame."""
    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError("Total mass must be positive")
    return positions.T @ masses / total_mass


def remove_com(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove translational COM motion from positions and velocities."""
    com = center_of_mass(positions, masses)
    positions_centered = positions - com

    momentum = velocities * masses[:, None]
    com_vel = momentum.sum(axis=0) / masses.sum()
    velocities_centered = velocities - com_vel
    return positions_centered, velocities_centered


@dataclass
class TrajectoryBatch:
    positions: np.ndarray  # shape (T, N, 3) in nanometers
    velocities: np.ndarray  # shape (T, N, 3) in nm/ps
    box: np.ndarray  # shape (T, 3, 3)
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    forces: Optional[np.ndarray]
    masses: np.ndarray
    atom_types: np.ndarray
    time_ps: np.ndarray
    metadata: Dict


def compute_time_grid(num_frames: int, dt_fs: float, save_interval: int) -> np.ndarray:
    """Return the simulation time (ps) for each stored frame."""
    times_fs = np.arange(num_frames) * dt_fs * save_interval
    return times_fs * 0.001


def to_device(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move numpy arrays in a batch dictionary to a torch device."""
    if torch is None:
        raise ImportError("PyTorch is required for training")
    return {key: torch.as_tensor(val, device=device) for key, val in batch.items()}


class ExponentialMovingAverage:
    """Simple scalar EMA helper for logging moving metrics."""

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value


@contextmanager
def numpy_print_options(**kwargs) -> Iterator[None]:
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def flatten_batch_dict(batch: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    """Stack selected batch dictionary entries along axis=0."""
    arrays = [batch[key] for key in keys]
    return np.stack(arrays, axis=0)
