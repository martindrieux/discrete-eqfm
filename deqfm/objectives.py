"""The two training objectives, as functions  loss_fn(model, raw, tokens) -> (loss, logs).

`model` is the (possibly DDP-wrapped) network used for every forward pass that receives a gradient.
`raw` is the same network unwrapped; it is used to build targets, without gradient.
"""
import torch

from .losses import (TARGETS, diagonal_data_loss, jvp_forward, make_jvp_fd, sample_diag_time, sample_jumps,
                     sample_state, soft_cross_entropy, uniform01)
from .nets import make_fn, probs_fn
from .reparam import get_reparam


def _reparam(cfg, vocab_size):
    L = cfg.loss
    if cfg.model.cond != "duration" or L.diag_time == "reparam" or L.jump_sampling == "reparam":
        return get_reparam(vocab_size)
    return None


def make_field_loss(cfg, vocab_size, clock):
    """Denoiser only (kappa = 0), trained with the weighted cross-entropy against the data."""
    L = cfg.loss
    rep = _reparam(cfg, vocab_size)

    def loss_fn(model, raw, tokens):
        t, w = sample_diag_time(tokens.shape[0], L.diag_time, L.t_eps, tokens.device, rep, L.stratified)
        x = sample_state(tokens, vocab_size, t)
        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        loss = diagonal_data_loss(fn(x, t, torch.zeros_like(t)), tokens, w)
        return loss, {"loss": loss.item()}

    return loss_fn


def make_flow_map_loss(cfg, vocab_size, clock, teacher=None, teacher_cond=None):
    """Algorithm 1: diagonal loss + alpha * off-diagonal loss.

    teacher=None : training from data. The diagonal uses the data and Dbar is the model's own denoiser.
    teacher given: distillation. The diagonal matches the frozen teacher and Dbar is the teacher.

    The two terms use the two halves of the batch (`split_batch`) and one forward pass.

    `boundary_prob` > 0 adds the boundary term: on that fraction of the off-diagonal samples the state is taken
    close to the data (t > boundary_t) and the target for a jump of any length, in particular the whole jump
    kappa = 1, is Dbar(x) (`boundary_target: denoiser`) or the data (`data`). A time-free network needs it: the
    semigroup target for a jump to the end is the model's own jump to the end from a later state, which nothing
    else ties down close to the data.
    """
    L = cfg.loss
    rep = _reparam(cfg, vocab_size)
    teacher_cond = teacher_cond or cfg.model.cond
    target_fn = TARGETS[L.off_diagonal]
    forward_mode = L.jvp == "forward"
    jvp = jvp_forward if forward_mode else make_jvp_fd(L.fd_eps)
    need_jvp = L.off_diagonal != "semigroup"

    def loss_fn(model, raw, tokens):
        B, dev = tokens.shape[0], tokens.device
        if L.split_batch and B >= 2:
            tok_diag, tok_off = tokens[:B // 2], tokens[B // 2:]
        else:
            tok_diag = tok_off = tokens
        nd, no = tok_diag.shape[0], tok_off.shape[0]
        f = probs_fn(make_fn(raw, cfg.model.cond, clock, vocab_size, **({"manual_attn": True} if need_jvp and forward_mode else {})))
        if teacher is None:
            Dbar = lambda y, t: f(y, t, torch.zeros_like(t))
        else:
            D = probs_fn(make_fn(teacher, teacher_cond, clock, vocab_size))
            Dbar = lambda y, t: D(y, t, torch.zeros_like(t))

        # diagonal (kappa = 0)
        t_d, w_d = sample_diag_time(nd, L.diag_time, L.t_eps, dev, rep, L.stratified)
        x_d = sample_state(tok_diag, vocab_size, t_d)

        # off-diagonal (kappa > 0)
        t_o, k3, k1 = sample_jumps(no, L.jump_sampling, clock, dev, L.off_diagonal == "semigroup", rep,
                                   L.whole_jump_prob, L.stratified)
        near = torch.zeros(no, dtype=torch.bool, device=dev)
        if L.boundary_prob > 0:
            near = torch.rand(no, device=dev) < L.boundary_prob
            t_o = torch.where(near, L.boundary_t + (1 - L.boundary_t) * torch.rand(no, device=dev), t_o)
            u = torch.rand(no, device=dev)
            k3 = torch.where(near, torch.where(u < 0.5, torch.ones_like(u), 2 * u - 1), k3)
        x_o = sample_state(tok_off, vocab_size, t_o)
        with torch.no_grad():
            far = ~near
            target = torch.empty_like(x_o)
            if far.any():
                target[far] = target_fn(f, Dbar, x_o[far], t_o[far], k3[far], None if k1 is None else k1[far],
                                        clock=clock, jvp=jvp)
            if near.any():
                if L.boundary_target == "data":
                    target[near] = torch.nn.functional.one_hot(tok_off[near], vocab_size).float()
                else:
                    target[near] = Dbar(x_o[near], t_o[near])
            if teacher is not None:
                target_d = Dbar(x_d, t_d)

        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        logits = fn(torch.cat([x_d, x_o]), torch.cat([t_d, t_o]), torch.cat([torch.zeros_like(t_d), k3]))
        if teacher is None:
            loss_diag = diagonal_data_loss(logits[:nd], tok_diag, w_d)
        else:
            loss_diag = soft_cross_entropy(logits[:nd], target_d)
        loss_off = soft_cross_entropy(logits[nd:], target)

        loss = loss_diag + L.alpha * loss_off
        return loss, {"loss": loss.item(), "diag": loss_diag.item(), "off": loss_off.item()}

    return loss_fn
