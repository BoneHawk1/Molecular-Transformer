import numpy as np
import torch

from kstep.data import KStepData, build_windows, save_dataset


def _traj(n_frames=30, n_atoms=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32),
        "vel": rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32),
        "masses": np.array([12.0, 1.0, 16.0, 1.0][:n_atoms], np.float32),
        "atom_types": np.array([6, 1, 8, 1][:n_atoms]),
    }


def test_round_trip_and_gather(tmp_path):
    rng = np.random.default_rng(0)
    t1, t2 = _traj(seed=1), _traj(n_atoms=3, seed=2)
    parts = [build_windows(t1, 4, 2, 0, rng), build_windows(t2, 4, 1, 5, rng)]
    path = tmp_path / "ds.npz"
    save_dataset(path, parts, ["a", "b"], 4, 0.1)
    ds = KStepData(path)
    assert len(ds) == len(parts[0]["n_atoms"]) + 5
    assert ds.k_steps == 4 and abs(ds.frame_dt_ps - 0.1) < 1e-12
    only_b = KStepData(path, ["b"])
    assert len(only_b) == 5 and only_b.molecules == ["b"]
    b = ds.gather(torch.tensor([0, len(ds) - 1]))
    assert b["x_t"].shape == (4 + 3, 3)
    assert b["batch"].tolist() == [0] * 4 + [1] * 3
    # first sample of molecule a: window starting at frame 0, COM removed
    x0 = t1["pos"][0] - (t1["pos"][0].T @ t1["masses"]) / t1["masses"].sum()
    np.testing.assert_allclose(b["x_t"][:4].numpy(), x0, atol=1e-5)
    np.testing.assert_allclose(b["masses"][:4].numpy(), t1["masses"])


def test_normalization_tables(tmp_path):
    rng = np.random.default_rng(0)
    save_dataset(tmp_path / "ds.npz", [build_windows(_traj(), 2, 1, 0, rng)], ["a"], 2, 0.1)
    tables = KStepData(tmp_path / "ds.npz").normalization_tables(10)
    for t in tables.values():
        assert t.shape == (11,) and torch.all(t > 0)
    assert tables["vel_scale"][6] != tables["vel_scale"][1]
