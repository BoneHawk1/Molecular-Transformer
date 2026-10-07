import numpy as np
import torch

from kstep.common import (ASE_FS, ase_velocity_to_nm_per_ps, friction_per_ps_to_ase, nm_per_ps_to_ase_velocity)
from kstep.hybrid import CorrectorResult, HybridOptions, ZeroPredictor, run_hybrid
from kstep.model import KStepModel, ModelConfig


class HarmonicCorrector:
    """Toy corrector: velocity-Verlet on springs between consecutive atoms."""

    micro_dt_ps = 0.002

    def __init__(self, ref):
        self.ref = ref
        self.calls = 0

    def run(self, pos, vel, steps):
        self.calls += 1
        pos, vel = pos.copy(), vel.copy()
        for _ in range(steps):
            force = -(pos - self.ref) * 50.0
            vel += 0.5 * self.micro_dt_ps * force / 12.0
            pos += self.micro_dt_ps * vel
            force = -(pos - self.ref) * 50.0
            vel += 0.5 * self.micro_dt_ps * force / 12.0
        return CorrectorResult(pos, vel, float(0.5 * 12 * (vel ** 2).sum()), 0.0)


class NaNPredictor:
    stochastic = False

    def predict(self, batch, generator=None, **_):
        return torch.full_like(batch["x_t"], float("nan")), torch.zeros_like(batch["v_t"])


def _setup():
    ref = np.array([[0, 0, 0], [0.15, 0, 0], [0.2, 0.14, 0]], float)
    return ref, np.zeros_like(ref), np.full(3, 12.0), np.array([6, 6, 6])


def test_zero_predictor_accounting():
    ref, vel, m, z = _setup()
    opts = HybridOptions(jump_time_ps=0.4, corrector_steps=10)
    out = run_hybrid(ZeroPredictor(), HarmonicCorrector(ref), ref, vel, m, z, 20, opts)
    s = out["summary"]
    assert out["pos"].shape == (21, 3, 3)
    assert s["force_calls"] == 200 and s["force_calls_reference"] == 20 * (200 + 10)
    assert abs(s["force_call_savings"] - 21.0) < 1e-9
    assert s["accepted_fraction"] == 1.0
    np.testing.assert_allclose(out["time_ps"][1], 0.42)


def test_rejected_proposals_fall_back_to_reference():
    ref, vel, m, z = _setup()
    opts = HybridOptions(jump_time_ps=0.4, corrector_steps=10, max_attempts=2)
    out = run_hybrid(NaNPredictor(), HarmonicCorrector(ref), ref, vel, m, z, 5, opts)
    assert out["summary"]["fallback_fraction"] == 1.0
    assert np.all(out["attempts"] == 2)
    assert np.isfinite(out["pos"]).all()


def test_uq_escalation():
    ref, vel, m, z = _setup()
    model = KStepModel(ModelConfig(kind="flow", hidden_dim=16, num_layers=1, num_rbf=4, flow_steps=2)).eval()
    opts = HybridOptions(jump_time_ps=0.4, corrector_steps=5, uq_samples=3, uq_threshold=0.0)
    out = run_hybrid(model, HarmonicCorrector(ref), ref, vel + 0.1, m, z, 4, opts)
    assert out["summary"]["escalated_fraction"] == 1.0
    assert np.isfinite(out["uq_spread"]).all()


def test_ase_unit_conversions():
    v = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(ase_velocity_to_nm_per_ps(nm_per_ps_to_ase_velocity(v)), v)
    # 1 Å/fs = 100 nm/ps; in ASE units 1 Å/fs = 1/ASE_FS
    np.testing.assert_allclose(ase_velocity_to_nm_per_ps(1.0 / ASE_FS), 100.0)
    np.testing.assert_allclose(friction_per_ps_to_ase(1000.0), 1.0 / ASE_FS)
