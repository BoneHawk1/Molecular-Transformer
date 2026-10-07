"""Backwards-compatible helpers for the numbered scripts.

Shared logic lives in :mod:`kstep.common`; this module re-exports it and adds the
few script-only helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from kstep.common import (LOGGER, center_of_mass, configure_logging, ensure_dir, kinetic_energy,  # noqa: F401
                          load_trajectory, load_yaml, remove_com, set_seed, write_json)


def read_smiles(path: Path) -> List[str]:
    """Read a SMILES file, skipping blank/commented lines."""
    smiles: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles.append(line)
    return smiles


def compute_time_grid(num_frames: int, dt_fs: float, save_interval: int) -> np.ndarray:
    """Simulation time (ps) of each stored frame."""
    return np.arange(num_frames) * dt_fs * save_interval * 0.001
