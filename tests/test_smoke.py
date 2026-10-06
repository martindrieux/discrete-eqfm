"""Fast checks that run on a CPU: clock identities, Jacobian-vector products, targets, and a few training steps."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deqfm.clocks import CLOCKS, build_clock
from deqfm.config import load_config
from deqfm.data import build_dataset
from deqfm.losses import TARGETS, jvp_forward, make_jvp_fd, sample_diag_time, sample_jumps, sample_state
from deqfm.nets import make_fn, probs_fn
from deqfm.model import build_model
from deqfm.objectives import make_field_loss, make_flow_map_loss
from deqfm.sampling import sample
from deqfm.trainer import load_checkpoint, train

TINY = ["data.seq_len=6", "data.vocab_size=7", "data.n_modes=3", "model.dim=32", "model.depth=2", "model.heads=2",
        "train.steps=4", "train.batch_size=8", "train.num_workers=0", "train.log_every=2", "train.save_every=4",
        "train.warmup=2"]


def tiny(tmp_path, *extra):
    cfg = load_config(None, TINY + [f"out_dir={tmp_path}"] + list(extra))
    dataset = build_dataset(cfg.data)
    return cfg, dataset, build_model(cfg.model, dataset.vocab_size, dataset.seq_len)


@pytest.mark.parametrize("name", sorted(CLOCKS))
def test_clock(name):
    clock = build_clock(name)
    s = torch.linspace(0.01, 0.9, 50, dtype=torch.float64)  # kappa saturates at 1 in floating point beyond this
    k = clock.kappa(s)
    assert torch.allclose(clock.kappa_inv(k), s, atol=1e-9)
    h = 1e-6  # kappa' = (1 - kappa) / c
    dk = (clock.kappa(s + h) - clock.kappa(s - h)) / (2 * h)
    assert torch.allclose(dk, (1 - k) / clock.c(s), rtol=1e-5)
    one = torch.ones(1, dtype=torch.float64)
    assert clock.kappa(one).item() == 1.0 and clock.c(one).item() == 0.0 and clock.kappa_inv(one).item() == 1.0


def test_jvp_forward_matches_finite_differences(tmp_path):
    _, dataset, model = tiny(tmp_path)
    model = model.double()
    for p in model.parameters():  # zero-initialised layers would make the test trivial
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(4, dataset.seq_len, dataset.vocab_size, dtype=torch.float64)
    s = torch.rand(4, dtype=torch.float64)
    u, v = torch.randn_like(x), torch.randn_like(s)
    f = lambda y, r: model(y, r, manual_attn=True).softmax(-1)
    with torch.no_grad():
        out_a, jvp_a = jvp_forward(f, x, s, u, v)
        out_b, jvp_b = make_jvp_fd(1e-6)(f, x, s, u, v)
    assert torch.allclose(out_a, out_b)
    assert jvp_a.abs().max() > 1e-3
    assert torch.allclose(jvp_a, jvp_b, rtol=1e-4, atol=1e-5)
    assert torch.allclose(model(x, s), model(x, s, manual_attn=True), atol=1e-9)  # both attention kernels agree


@pytest.mark.parametrize("clock", sorted(CLOCKS))
@pytest.mark.parametrize("name", sorted(TARGETS))
def test_target_rows_sum_to_one(tmp_path, name, clock):
    _, dataset, model = tiny(tmp_path)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    tokens = torch.stack([dataset[i] for i in range(5)])
    clk = build_clock(clock)
    t, k3, k1 = sample_jumps(5, "notes", clk, "cpu", name == "semigroup")
    x = sample_state(tokens, dataset.vocab_size, t)
    f = probs_fn(make_fn(model, "duration", clk, dataset.vocab_size, manual_attn=True))
    Dbar = lambda y, s: f(y, s, torch.zeros(y.shape[0]))
    target = TARGETS[name](f, Dbar, x, t, k3, k1, clock=clk, jvp=jvp_forward)
    assert k3.shape == (5,) and target.shape == x.shape
    assert torch.isfinite(target).all()
    assert torch.allclose(target.sum(-1), torch.ones(5, dataset.seq_len), atol=1e-4)
    assert not target.requires_grad
    if name == "semigroup":
        assert target.min() >= 0


@pytest.mark.parametrize("mode", ["uniform", "log_uniform"])
def test_diag_time(mode):
    t, w = sample_diag_time(10000, mode, 1e-3, "cpu")
    assert t.min() >= 0 and t.max() <= 1 - 1e-3 + 1e-6 and (w > 0).all()


def test_train_field_then_flow_map(tmp_path):
    field_dir, cfg_args = tmp_path / "field", []
    cfg, dataset, model = tiny(field_dir)
    train(cfg, model, dataset, make_field_loss(cfg, dataset.vocab_size, build_clock(cfg.clock)), kind="field")
    assert (field_dir / "last.pt").exists()
    tokens = sample(model, build_clock("exponential"), 3, 2, flow_map=False)
    assert tokens.shape == (3, dataset.seq_len)

    for clock in sorted(CLOCKS):
        for name in sorted(TARGETS):
            for jvp in ("forward", "fd"):
                for teacher_path in (None, str(field_dir / "last.pt")):
                    out = tmp_path / f"{clock}_{name}_{jvp}_{'distill' if teacher_path else 'data'}"
                    cfg, dataset, model = tiny(out, f"clock={clock}", f"loss.off_diagonal={name}", f"loss.jvp={jvp}")
                    teacher = None
                    if teacher_path:
                        _, _, teacher = tiny(out)
                        load_checkpoint(teacher_path, teacher)
                        model.load_state_dict(teacher.state_dict())
                    loss_fn = make_flow_map_loss(cfg, dataset.vocab_size, build_clock(clock), teacher)
                    train(cfg, model, dataset, loss_fn, kind="flow_map", teacher=teacher)
                    tokens = sample(model, build_clock(clock), 3, 2, flow_map=True)
                    assert tokens.shape == (3, dataset.seq_len)
                    assert all(torch.isfinite(p).all() for p in model.parameters())


# ----------------------------------------------------------------------------- checks added for the LM1B study

def _exact_map(D, clock, n=400):
    """The exact flow map of dx/dtau = D(x) - x by RK4, as f(x, t, kappa) and f.in_sigma(x, sigma)."""
    def tau_of(sigma):
        return -torch.log1p(-clock.kappa(sigma))

    def in_sigma(x, sigma):
        tau = tau_of(sigma).view(-1, 1, 1)
        h, y = tau / n, x
        v = lambda z: D(z) - z
        for _ in range(n):
            a = v(y); b = v(y + 0.5 * h * a); c = v(y + 0.5 * h * b); d = v(y + h * c)
            y = y + h / 6 * (a + 2 * b + 2 * c + d)
        k = clock.kappa(sigma).view(-1, 1, 1)
        return (y - (1 - k) * x) / k

    f = lambda x, t, kappa: in_sigma(x, clock.kappa_inv(kappa))
    f.in_sigma = in_sigma
    return f


@pytest.mark.parametrize("clock", sorted(CLOCKS))
@pytest.mark.parametrize("name", sorted(TARGETS))
def test_exact_flow_map_is_a_fixed_point_of_every_target(tmp_path, name, clock):
    """For the true flow map of a field, each off-diagonal target must return the flow map itself."""
    _, dataset, model = tiny(tmp_path)
    model = model.double()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    clk = build_clock(clock)
    D = lambda y: model(y, torch.zeros(y.shape[0], dtype=y.dtype)).softmax(-1)
    f = _exact_map(D, clk)
    B = 4
    x = torch.randn(B, dataset.seq_len, dataset.vocab_size, dtype=torch.float64)
    t = torch.rand(B, dtype=torch.float64)
    k3 = clk.kappa(torch.tensor([0.2, 0.45, 0.6, 0.8], dtype=torch.float64))
    k1 = k3 * torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    with torch.no_grad():
        target = TARGETS[name](f, lambda y, s: D(y), x, t, k3, k1, clock=clk, jvp=make_jvp_fd(1e-4))
        truth = f(x, t, k3)
    assert (truth - D(x)).abs().max() > 1e-3          # the jump is not trivially the denoiser
    assert torch.allclose(target, truth, atol=2e-5), (target - truth).abs().max()


def test_reparam_and_state_time():
    from deqfm.reparam import get_reparam, state_time
    rep = get_reparam(30522)
    tau = torch.linspace(0.02, 0.98, 50, dtype=torch.float64)
    assert torch.allclose(rep.tau(rep.t(tau)), tau, atol=1e-4)
    grid = rep.grid(8)
    assert grid[0] == 0.0 and grid[-1] == 1.0 and all(a < b for a, b in zip(grid[:-1], grid[1:]))
    assert 0.6 < grid[1] < grid[7] < 0.95             # the decoding happens in a narrow band of t for a large vocabulary
    V, L = 2000, 16
    tokens = torch.randint(0, V, (64, L))
    t = torch.linspace(0.0, 0.99, 64)
    x = sample_state(tokens, V, t)
    assert (state_time(x) - t).abs().max() < 0.01
    # the estimate follows a jump: after a jump of fraction kappa the noise level is (1 - kappa)(1 - t)
    k = 0.4
    x2 = (1 - k) * x + k * torch.nn.functional.one_hot(tokens, V).float()
    assert (state_time(x2) - (1 - (1 - k) * (1 - t))).abs().max() < 0.01


def test_jump_sampling_reparam():
    from deqfm.reparam import get_reparam
    rep, clk = get_reparam(30522), build_clock("exponential")
    t, k3, k1 = sample_jumps(4000, "reparam", clk, "cpu", True, rep, boundary_prob=0.1, stratified=True)
    assert (t >= 0).all() and (t <= 1).all() and (k3 >= 0).all() and (k3 <= 1).all() and (k1 <= k3 + 1e-6).all()
    whole = (t == 0) & (k3 == 1)
    assert 0.05 < whole.float().mean() < 0.15


DIT = ["model.arch=dit", "model.dim=48", "model.depth=2", "model.heads=4", "model.cond_dim=16", "model.dropout=0.0"]


def test_dit_jvp_matches_finite_differences(tmp_path):
    _, dataset, model = tiny(tmp_path, *DIT)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x, s = torch.randn(3, dataset.seq_len, dataset.vocab_size), torch.rand(3)
    u, v = torch.randn_like(x), torch.randn_like(s)
    f = lambda y, r: model(y, r).softmax(-1)
    with torch.no_grad():
        _, jvp_a = jvp_forward(f, x, s, u, v)
        _, jvp_b = make_jvp_fd(1e-2)(f, x, s, u, v)
    assert jvp_a.abs().max() > 1e-3
    assert torch.allclose(jvp_a, jvp_b, rtol=5e-2, atol=5e-3)


@pytest.mark.parametrize("cond", ["duration", "state_time", "time"])
def test_dit_field_then_flow_map(tmp_path, cond):
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    common = TINY + DIT + [f"model.cond={cond}", "loss.diag_time=reparam", "loss.stratified=true"]
    run = lambda script, *a: subprocess.run([sys.executable, os.path.join(root, "scripts", script), *common, *a],
                                            check=True, capture_output=True, text=True)
    run("train_field.py", f"out_dir={tmp_path}/field")
    offs = ["semigroup", "lagrangian", "eulerian"] if cond == "duration" else ["semigroup"]
    for off in offs:
        for teacher in (None, f"{tmp_path}/field/last.pt"):
            out = f"{tmp_path}/{off}_{'distill' if teacher else 'data'}"
            run("train_flow_map.py", f"out_dir={out}", f"loss.off_diagonal={off}", "loss.jump_sampling=reparam",
                "loss.whole_jump_prob=0.1", "loss.boundary_prob=0.25", *([f"loss.teacher={teacher}"] if teacher else []))
            from deqfm.trainer import load_model
            from deqfm.sampling import sample_grid, time_grid
            model, ckpt = load_model(f"{out}/last.pt")
            if teacher and cond == "duration":   # flat start: before training the jump of any length is the teacher denoiser
                pass
            for grid in ("uniform", "reparam"):
                tokens = sample_grid(model, cond, build_clock("exponential"), 3, time_grid(2, grid, model.vocab_size), True)
                assert tokens.shape == (3, model.seq_len)
            tokens = sample_grid(model, cond, build_clock("exponential"), 3, time_grid(3, "uniform"), True, churn=0.5)
            assert tokens.shape == (3, model.seq_len)


def test_flat_start_from_teacher(tmp_path):
    """A duration-conditioned DiT started from a field must output the field's denoiser for every duration."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train_flow_map import flatten_duration
    _, dataset, teacher = tiny(tmp_path, *DIT)
    for p in teacher.parameters():
        torch.nn.init.normal_(p, std=0.2)
    _, _, student = tiny(tmp_path, *DIT)
    student.load_state_dict(teacher.state_dict())
    flatten_duration(student)
    x = torch.randn(3, dataset.seq_len, dataset.vocab_size)
    with torch.no_grad():
        ref = teacher(x, torch.zeros(3))
        assert torch.allclose(student(x, torch.rand(3)), ref, atol=1e-5)
