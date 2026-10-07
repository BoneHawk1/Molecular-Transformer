"""Bond perception and vectorised internal-coordinate measurements (numpy)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Covalent radii in Å (Cordero et al. 2008) for the elements we handle.
COVALENT_RADII: Dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}


def guess_bonds(positions_nm: np.ndarray, atom_types: np.ndarray, tol_angstrom: float = 0.1) -> np.ndarray:
    """Return bonded pairs ``(B, 2)`` with i < j using covalent radii + tolerance."""
    coords = np.asarray(positions_nm, dtype=np.float64) * 10.0
    radii = np.array([COVALENT_RADII.get(int(z), 0.75) for z in atom_types])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    cutoff = radii[:, None] + radii[None, :] + tol_angstrom
    ii, jj = np.where(np.triu(dist <= cutoff, k=1))
    return np.stack([ii, jj], axis=1).astype(np.int64) if len(ii) else np.zeros((0, 2), dtype=np.int64)


def _adjacency(bonds: np.ndarray) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for i, j in bonds.tolist():
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    return adj


def build_angles(bonds: np.ndarray) -> np.ndarray:
    """Angle triples ``(A, 3)`` (i, centre, k) from the bond graph."""
    triples: List[Tuple[int, int, int]] = []
    for centre, neigh in _adjacency(bonds).items():
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                triples.append((neigh[a], centre, neigh[b]))
    return np.asarray(triples, dtype=np.int64).reshape(-1, 3)


def build_dihedrals(bonds: np.ndarray, heavy_mask: np.ndarray | None = None) -> np.ndarray:
    """Unique proper dihedrals ``(D, 4)``.

    If ``heavy_mask`` is given, only dihedrals whose two central atoms are heavy
    atoms are returned (these are the conformationally interesting torsions).
    """
    adj = _adjacency(bonds)
    seen = set()
    quads: List[Tuple[int, int, int, int]] = []
    for j, k in bonds.tolist():
        if heavy_mask is not None and not (heavy_mask[j] and heavy_mask[k]):
            continue
        for i in adj.get(j, []):
            if i == k:
                continue
            for l in adj.get(k, []):
                if l in (i, j):
                    continue
                key = (i, j, k, l)
                if key in seen or key[::-1] in seen:
                    continue
                seen.add(key)
                quads.append(key)
    return np.asarray(quads, dtype=np.int64).reshape(-1, 4)


def bond_lengths(pos: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """Bond lengths in Å. ``pos`` is ``(..., N, 3)`` in nm; returns ``(..., B)``."""
    pos = np.asarray(pos, dtype=np.float64) * 10.0
    return np.linalg.norm(pos[..., bonds[:, 0], :] - pos[..., bonds[:, 1], :], axis=-1)


def angles_deg(pos: np.ndarray, triples: np.ndarray) -> np.ndarray:
    """Valence angles in degrees, ``(..., A)``."""
    pos = np.asarray(pos, dtype=np.float64)
    v1 = pos[..., triples[:, 0], :] - pos[..., triples[:, 1], :]
    v2 = pos[..., triples[:, 2], :] - pos[..., triples[:, 1], :]
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def dihedrals_deg(pos: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Signed dihedral angles in degrees in (-180, 180], ``(..., D)``."""
    pos = np.asarray(pos, dtype=np.float64)
    p0, p1, p2, p3 = (pos[..., quads[:, n], :] for n in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def kabsch_rmsd(ref: np.ndarray, mob: np.ndarray, masses: np.ndarray | None = None) -> float:
    """Optimal-superposition RMSD between two ``(N, 3)`` frames (same units as input)."""
    w = np.ones(len(ref)) if masses is None else np.asarray(masses, dtype=np.float64)
    w = w / w.sum()
    a = ref - (w[:, None] * ref).sum(0)
    b = mob - (w[:, None] * mob).sum(0)
    h = (b * w[:, None]).T @ a
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    diff = b @ rot - a
    return float(np.sqrt((w * np.sum(diff ** 2, axis=1)).sum()))
