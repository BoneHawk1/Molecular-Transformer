"""Small helpers shared by every stage of the pipeline."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

LOGGER = logging.getLogger("md_kstep")

# Boltzmann constant in kJ/(mol K). kT / mass(g/mol) has units (nm/ps)^2.
KB_KJ_PER_MOL_K = 0.0083144626


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a terse logging format for CLI scripts."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def load_yaml(path: Path) -> Dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(data: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    total_mass = float(np.sum(masses))
    if total_mass <= 0:
        raise ValueError("Total mass must be positive")
    return positions.T @ masses / total_mass


def remove_com(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove centre-of-mass position and momentum from a single frame."""
    com = center_of_mass(positions, masses)
    com_vel = (velocities * masses[:, None]).sum(axis=0) / masses.sum()
    return positions - com, velocities - com_vel


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Kinetic energy in kJ/mol for velocities in nm/ps and masses in g/mol.

    Works on a single frame ``(N, 3)`` or a trajectory ``(T, N, 3)``.
    """
    return 0.5 * np.sum(masses[..., :, None] * velocities ** 2, axis=(-1, -2))


def load_trajectory(path: Path) -> Dict:
    """Load a trajectory NPZ written by 01/01b/06/06b into a plain dict."""
    data = np.load(path, allow_pickle=True)
    out: Dict = {key: data[key] for key in data.files}
    for key in ("metadata", "nve_windows"):
        if key in out:
            try:
                out[key] = json.loads(str(out[key]))
            except (TypeError, json.JSONDecodeError):
                pass
    out.setdefault("metadata", {})
    out.setdefault("nve_windows", [])
    meta = out["metadata"] if isinstance(out["metadata"], dict) else {}
    is_ase_qm = "qm_method" in meta or "basis" in meta
    if is_ase_qm and "velocity_units" not in meta and "vel" in out:
        # Written by the pre-Oct-2026 QM scripts: velocities are v_ase * 100, not nm/ps.
        LOGGER.warning("%s: rescaling legacy QM velocities by ase.units.fs (%.4f) to nm/ps", path, ASE_FS)
        out["vel"] = out["vel"] * ASE_FS
        meta["velocity_units"] = "nm/ps (rescaled on load)"
    return out


# ---------------------------------------------------------------------------
# ASE unit conversions. ASE velocities are in Å per *ASE time unit*
# (1 fs = ase.units.fs ≈ 0.0982 ASE time units), not Å/fs, and Langevin friction
# is in inverse ASE time units. The 2025 QM scripts treated both as per-fs, which
# stored QM velocities ≈10.2× too large and made the friction ≈100× too small.
# ---------------------------------------------------------------------------
ASE_FS = 0.09822694750253276  # == ase.units.fs (checked at import time when ASE is installed)
try:  # pragma: no cover - depends on optional ASE install
    from ase import units as _ase_units

    ASE_FS = float(_ase_units.fs)
except ImportError:  # pragma: no cover
    pass


def ase_velocity_to_nm_per_ps(v_ase: np.ndarray) -> np.ndarray:
    return np.asarray(v_ase) * ASE_FS * 100.0


def nm_per_ps_to_ase_velocity(v_nm_ps: np.ndarray) -> np.ndarray:
    return np.asarray(v_nm_ps) / (ASE_FS * 100.0)


def friction_per_ps_to_ase(friction_per_ps: float) -> float:
    return friction_per_ps / (1000.0 * ASE_FS)
