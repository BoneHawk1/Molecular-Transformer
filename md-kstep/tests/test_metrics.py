import numpy as np

from kstep.geometry import dihedrals_deg, guess_bonds, kabsch_rmsd
from kstep.metrics import Topology, compare, js_divergence


def _butane_like(n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([[0, 0, 0], [0.15, 0, 0], [0.2, 0.14, 0], [0.35, 0.14, 0.02]], float)
    traj = base + rng.normal(scale=0.004, size=(n_frames, 4, 3))
    return traj, np.array([6, 6, 6, 6])


def test_js_bounds():
    assert js_divergence([1, 2, 3], [1, 2, 3]) < 1e-9
    assert abs(js_divergence([1, 0], [0, 1]) - 1.0) < 1e-6


def test_dihedral_sign_and_value():
    p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], float)
    assert abs(abs(dihedrals_deg(p, np.array([[0, 1, 2, 3]]))[0]) - 90.0) < 1e-6
    mirrored = p * np.array([1, -1, 1])
    assert np.sign(dihedrals_deg(p, np.array([[0, 1, 2, 3]]))[0]) != np.sign(dihedrals_deg(mirrored, np.array([[0, 1, 2, 3]]))[0])


def test_kabsch_rotation_invariant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert kabsch_rmsd(a, a @ q.T + 3.0) < 1e-9


def test_compare_identical_is_perfect():
    traj, z = _butane_like()
    topo = Topology(traj[0], z)
    assert len(topo.bonds) == 3 and len(topo.torsions) == 1
    vel = np.random.default_rng(1).normal(size=traj.shape)
    m = compare({"pos": traj, "vel": vel}, {"pos": traj, "vel": vel}, topo, 0.4, 0.4, np.full(4, 12.0))
    assert m["bond_js"] < 1e-9 and m["torsion_js"] < 1e-9
    assert abs(m["lag_rmsd_A_ratio"] - 1) < 1e-9 and abs(m["ekin_ratio"] - 1) < 1e-9


def test_frozen_trajectory_has_low_lag_rmsd_ratio():
    traj, z = _butane_like()
    topo = Topology(traj[0], z)
    frozen = traj[:1] + 0.1 * (traj - traj[:1])
    m = compare({"pos": traj}, {"pos": frozen}, topo, 0.4, 0.4, np.full(4, 12.0))
    assert m["lag_rmsd_A_ratio"] < 0.2
    assert m["bond_js"] > 0.1
