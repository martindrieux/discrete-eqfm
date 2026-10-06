"""Fast checks that run on a CPU: clock identities, Jacobian-vector products, targets, and a few training steps."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deqfm.clocks import CLOCKS, build_clock
from deqfm.config import load_config
from deqfm.data import build_dataset
from deqfm.losses import TARGETS, jvp_forward, make_jvp_fd, sample_diag_time, sample_state
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
    x = sample_state(tokens, dataset.vocab_size, torch.rand(5))
    f = lambda y, r: model(y, r, manual_attn=True).softmax(-1)
    Dbar = lambda y: f(y, torch.zeros(y.shape[0]))
    sigma, target = TARGETS[name](f, Dbar, x, build_clock(clock), jvp_forward)
    assert sigma.shape == (5,) and target.shape == x.shape
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
    train(cfg, model, dataset, make_field_loss(cfg, dataset.vocab_size), kind="field")
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
