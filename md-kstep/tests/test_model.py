import pytest
import torch

from kstep.model import ModelConfig, KStepModel, full_graph, project_zero_com

torch.manual_seed(0)


def _batch(sizes=(5, 7), seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    z = torch.randint(1, 9, (n,), generator=g)
    return {
        "x_t": torch.randn(n, 3, generator=g) * 0.15,
        "v_t": torch.randn(n, 3, generator=g),
        "x_tk": torch.randn(n, 3, generator=g) * 0.15,
        "v_tk": torch.randn(n, 3, generator=g),
        "atom_types": z,
        "masses": z.float() * 2.0,
        "batch": torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)),
        "num_graphs": len(sizes),
    }


def _model(kind):
    torch.manual_seed(1)
    m = KStepModel(ModelConfig(kind=kind, hidden_dim=32, num_layers=2, num_rbf=8, flow_steps=3)).double().eval()
    return m


def _orthogonal(seed, reflect=False):
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if reflect:
        q = q @ torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    return q


def _double(b):
    return {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}


@pytest.mark.parametrize("kind", ["mean", "flow"])
@pytest.mark.parametrize("reflect", [False, True])
def test_rotation_equivariance(kind, reflect):
    m, b = _model(kind), _double(_batch())
    R = _orthogonal(3, reflect)
    noise = (torch.randn_like(b["x_t"]), torch.randn_like(b["v_t"]))
    dx, dv = m.predict(b, noise=noise)
    rb = dict(b, x_t=b["x_t"] @ R.T, v_t=b["v_t"] @ R.T)
    rdx, rdv = m.predict(rb, noise=(noise[0] @ R.T, noise[1] @ R.T))
    torch.testing.assert_close(rdx, dx @ R.T, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(rdv, dv @ R.T, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", ["mean", "flow"])
def test_translation_invariance_and_permutation(kind):
    m, b = _model(kind), _double(_batch(sizes=(6,)))
    noise = (torch.randn_like(b["x_t"]), torch.randn_like(b["v_t"]))
    dx, dv = m.predict(b, noise=noise)
    tdx, tdv = m.predict(dict(b, x_t=b["x_t"] + torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)), noise=noise)
    torch.testing.assert_close(tdx, dx)
    torch.testing.assert_close(tdv, dv)
    perm = torch.randperm(6)
    pb = {k: (v[perm] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 6 else v) for k, v in b.items()}
    pdx, pdv = m.predict(pb, noise=(noise[0][perm], noise[1][perm]))
    torch.testing.assert_close(pdx, dx[perm])
    torch.testing.assert_close(pdv, dv[perm])


@pytest.mark.parametrize("kind", ["mean", "flow"])
def test_momentum_conserved(kind):
    m, b = _model(kind), _double(_batch())
    dx, dv = m.predict(b)
    for d in (dx, dv):
        per_graph = torch.zeros(2, 3, dtype=d.dtype).index_add_(0, b["batch"], b["masses"].unsqueeze(-1) * d)
        assert per_graph.abs().max() < 1e-9


def test_flow_is_stochastic_mean_is_not():
    b = _double(_batch())
    flow, mean = _model("flow"), _model("mean")
    a1, _ = flow.predict(b, generator=torch.Generator().manual_seed(1))
    a2, _ = flow.predict(b, generator=torch.Generator().manual_seed(2))
    assert (a1 - a2).abs().max() > 1e-6
    m1, _ = mean.predict(b)
    m2, _ = mean.predict(b)
    torch.testing.assert_close(m1, m2)


def test_full_graph_matches_bruteforce():
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    ei = full_graph(batch)
    expected = {(i, j) for i in range(9) for j in range(9) if i != j and batch[i] == batch[j]}
    assert set(map(tuple, ei.t().tolist())) == expected
    assert ei.shape[1] == len(expected)


def test_projection_removes_weighted_sum():
    y = torch.randn(9, 3, dtype=torch.float64)
    w = torch.rand(9, dtype=torch.float64) + 0.1
    batch = torch.tensor([0] * 4 + [1] * 5)
    p = project_zero_com(y, w, batch, 2)
    s = torch.zeros(2, 3, dtype=torch.float64).index_add_(0, batch, w.unsqueeze(-1) * p)
    assert s.abs().max() < 1e-12


@pytest.mark.parametrize("kind", ["mean", "flow"])
def test_loss_backward(kind):
    m = KStepModel(ModelConfig(kind=kind, hidden_dim=16, num_layers=1, num_rbf=4))
    loss, parts = m.loss(_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_sde_sampler_is_equivariant_and_conserves_momentum():
    m, b = _model("flow"), _double(_batch())
    R = _orthogonal(5)
    g = torch.Generator().manual_seed(0)
    n0 = (torch.randn(b["x_t"].shape, generator=g, dtype=torch.float64), torch.randn(b["x_t"].shape, generator=g, dtype=torch.float64))
    sn = (torch.randn((3,) + tuple(b["x_t"].shape), generator=g, dtype=torch.float64),
          torch.randn((3,) + tuple(b["x_t"].shape), generator=g, dtype=torch.float64))
    kw = dict(solver="sde", sde_eps=1.0, sde_t_start=0.0)
    dx, dv = m.predict(b, noise=n0, step_noise=sn, **kw)
    rb = dict(b, x_t=b["x_t"] @ R.T, v_t=b["v_t"] @ R.T)
    rdx, rdv = m.predict(rb, noise=(n0[0] @ R.T, n0[1] @ R.T), step_noise=(sn[0] @ R.T, sn[1] @ R.T), **kw)
    torch.testing.assert_close(rdx, dx @ R.T, rtol=1e-6, atol=1e-9)
    per_graph = torch.zeros(2, 3, dtype=dx.dtype).index_add_(0, b["batch"], b["masses"].unsqueeze(-1) * dv)
    assert per_graph.abs().max() < 1e-9
