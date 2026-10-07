"""Evaluation metrics for hybrid trajectories.

Frame-by-frame comparison of two chaotic/stochastic trajectories (the 2025 metric) is
dominated by decorrelation after ~100 fs, so this module compares *distributions*
and *dynamics* instead:

Equilibrium (does the hybrid sample the right ensemble?)
  - Jensen–Shannon divergence of bond, angle and torsion histograms
  - L1 distance of the heavy-atom pair-distance distribution ("RDF")
  - mean kinetic energy ratio

Dynamics (does one macro-step move the system as far as k baseline frames do?)
  - lag-RMSD ratio: mean Kabsch RMSD between consecutive frames, hybrid / reference
  - torsion-step ratio: mean |Δφ| per macro-step, hybrid / reference
  - torsion transition rate between the three 120° wells, per ps

A corrector-only control (Δ = 0) samples the right equilibrium but moves far too
little per macro-step, so the dynamics metrics are what separate a useful model from
no model. A second baseline seed gives the noise floor for every number.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .common import kinetic_energy
from .geometry import angles_deg, bond_lengths, build_angles, build_dihedrals, dihedrals_deg, guess_bonds, kabsch_rmsd


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (base 2, in [0, 1]) between two histograms."""
    p = np.asarray(p, np.float64) + eps
    q = np.asarray(q, np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def mean_column_js(a: np.ndarray, b: np.ndarray, bins: int, value_range: Optional[tuple] = None) -> float:
    """Average JS divergence over columns of two ``(T, K)`` sample matrices."""
    if a.shape[1] == 0:
        return float("nan")
    out = []
    for k in range(a.shape[1]):
        lo, hi = value_range if value_range else (min(a[:, k].min(), b[:, k].min()), max(a[:, k].max(), b[:, k].max()))
        if hi <= lo:
            hi = lo + 1e-6
        ha, _ = np.histogram(a[:, k], bins=bins, range=(lo, hi))
        hb, _ = np.histogram(b[:, k], bins=bins, range=(lo, hi))
        out.append(js_divergence(ha, hb))
    return float(np.mean(out))


def pair_distance_hist(pos: np.ndarray, atom_types: np.ndarray, bins: np.ndarray) -> np.ndarray:
    heavy = np.where(np.asarray(atom_types) > 1)[0]
    if len(heavy) < 2:
        return np.zeros(len(bins) - 1)
    iu = np.triu_indices(len(heavy), k=1)
    p = pos[:, heavy] * 10.0
    d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)[:, iu[0], iu[1]]
    hist, _ = np.histogram(d.ravel(), bins=bins, density=True)
    return hist


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


WELL_CENTERS_DEG = np.array([-60.0, 60.0, 180.0])


def torsion_wells(phi_deg: np.ndarray, core_deg: float = 30.0) -> np.ndarray:
    """Assign each frame of each torsion to a staggered well (-60, 60, 180) with hysteresis.

    A torsion only changes well once it gets within ``core_deg`` of the new well's centre;
    in between it keeps its previous assignment. This avoids counting fluctuations across
    a boundary (e.g. around ±180 or around 0 for planar torsions) as transitions.
    Frames before the first assignment get -1. ``phi_deg`` is ``(T, K)``.
    """
    phi = np.asarray(phi_deg, dtype=np.float64)
    dist = np.abs(wrap_deg(phi[..., None] - WELL_CENTERS_DEG))      # (T, K, 3)
    nearest = dist.argmin(-1)
    in_core = dist.min(-1) < core_deg
    out = np.full(phi.shape, -1, dtype=np.int64)
    state = np.full(phi.shape[1:], -1, dtype=np.int64)
    for t in range(phi.shape[0]):
        state = np.where(in_core[t], nearest[t], state)
        out[t] = state
    return out


def count_transitions(wells: np.ndarray) -> int:
    """Number of well changes, ignoring frames that are not yet assigned (-1)."""
    a, b = wells[:-1], wells[1:]
    return int(((a != b) & (a >= 0) & (b >= 0)).sum())


class Topology:
    """Internal coordinates perceived once from a reference frame."""

    def __init__(self, ref_frame: np.ndarray, atom_types: np.ndarray) -> None:
        self.atom_types = np.asarray(atom_types)
        self.bonds = guess_bonds(ref_frame, self.atom_types)
        self.angles = build_angles(self.bonds)
        heavy = self.atom_types > 1
        self.torsions = build_dihedrals(self.bonds, heavy_mask=heavy)


def equilibrium_metrics(ref: np.ndarray, test: np.ndarray, topo: Topology,
                        ref_vel: Optional[np.ndarray] = None, test_vel: Optional[np.ndarray] = None,
                        masses: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "bond_js": mean_column_js(bond_lengths(ref, topo.bonds), bond_lengths(test, topo.bonds), 40),
        "angle_js": mean_column_js(angles_deg(ref, topo.angles), angles_deg(test, topo.angles), 40),
        "torsion_js": mean_column_js(dihedrals_deg(ref, topo.torsions), dihedrals_deg(test, topo.torsions), 36, (-180, 180)),
    }
    bins = np.linspace(0.0, 12.0, 121)
    out["rdf_l1"] = float(np.abs(pair_distance_hist(ref, topo.atom_types, bins) - pair_distance_hist(test, topo.atom_types, bins)).sum() * (bins[1] - bins[0]))
    if ref_vel is not None and test_vel is not None and masses is not None:
        out["ekin_ratio"] = float(kinetic_energy(test_vel, masses).mean() / kinetic_energy(ref_vel, masses).mean())
    return out


def dynamics_stats(traj: np.ndarray, topo: Topology, frame_ps: float, masses: Optional[np.ndarray] = None,
                   max_pairs: int = 400) -> Dict[str, float]:
    """Per-frame-step dynamics of a trajectory whose frames are ``frame_ps`` apart."""
    T = len(traj)
    idx = np.linspace(0, T - 2, num=min(max_pairs, T - 1)).astype(int)
    rmsd = np.mean([kabsch_rmsd(traj[i], traj[i + 1], masses) for i in idx]) * 10.0  # Å
    out = {"lag_rmsd_A": float(rmsd)}
    if len(topo.torsions):
        phi = dihedrals_deg(traj, topo.torsions)
        out["torsion_step_deg"] = float(np.abs(wrap_deg(np.diff(phi, axis=0))).mean())
        flips = count_transitions(torsion_wells(phi))
        out["torsion_transitions_per_ps"] = float(flips / ((T - 1) * frame_ps * len(topo.torsions)))
    else:
        out["torsion_step_deg"] = float("nan")
        out["torsion_transitions_per_ps"] = float("nan")
    return out


def compare(ref: Dict, test: Dict, topo: Topology, ref_frame_ps: float, test_frame_ps: float,
            masses: np.ndarray) -> Dict[str, float]:
    """All metrics for ``test`` against ``ref``; each holds ``pos``/``vel`` frames ``*_frame_ps`` apart."""
    eq = equilibrium_metrics(ref["pos"], test["pos"], topo, ref.get("vel"), test.get("vel"), masses)
    d_ref = dynamics_stats(ref["pos"], topo, ref_frame_ps, masses)
    d_test = dynamics_stats(test["pos"], topo, test_frame_ps, masses)
    out = dict(eq)
    for key, val in d_test.items():
        out[key] = val
        base = d_ref[key]
        out[f"{key}_ratio"] = float(val / base) if base and np.isfinite(base) and base != 0 else float("nan")
    return out
