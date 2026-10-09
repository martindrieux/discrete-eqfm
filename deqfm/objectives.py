"""The two training objectives, as functions  loss_fn(model, raw, tokens) -> (loss, logs).

`model` is the (possibly DDP-wrapped) network used for every forward pass that receives a gradient.
`raw` is the same network unwrapped; it is used to build targets, without gradient.
"""
import torch

import copy

from .losses import (TARGETS, diagonal_data_loss, jvp_forward, make_jvp_fd, ot_coupling, partial_context,
                     refinement_rollout, refinement_round_loss, sample_diag_time, sample_jumps, sample_state,
                     soft_cross_entropy, uniform01)
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

    Endpoint-only learning (the transport map of Discrete Beckmann Transport Models) is the special case
    `jump_sampling: endpoint`, `off_diagonal: eulerian`, `direction: data`, `diag_weight: 0`, with the boundary term
    on the data (`boundary_target: data`): the Eulerian target at kappa = 1 is delta + (x_1 - x) . grad delta, the
    conservation equation b . grad T = 0 with the data point standing in for the denoiser, and no denoiser is trained.

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
    if L.partial_context or L.ril_frac > 0:
        assert cfg.model.cond == "duration", "partial context and refinement need a network that is told no time"
    use_diag = L.diag_weight > 0
    if L.direction == "data":
        assert L.off_diagonal == "eulerian", "loss.direction = data is only defined for the Eulerian loss"

    def loss_fn(model, raw, tokens):
        B, dev = tokens.shape[0], tokens.device
        if not use_diag:
            tok_diag, tok_off = tokens[:0], tokens
        elif L.split_batch and B >= 2:
            tok_diag, tok_off = tokens[:B // 2], tokens[B // 2:]
        else:
            tok_diag = tok_off = tokens
        nd, no = tok_diag.shape[0], tok_off.shape[0]
        # network that builds the targets: the trained weights (as FMLM), or their running average, which moves slowly
        tnet = getattr(loss_fn, "ema", None) if L.target_net == "ema" else raw
        tnet = raw if tnet is None else tnet
        was_training = tnet.training
        if L.target_eval:        # no dropout in the targets
            tnet.eval()
        f = probs_fn(make_fn(tnet, cfg.model.cond, clock, vocab_size, **({"manual_attn": True} if need_jvp and forward_mode else {})))
        if teacher is None:
            Dbar = lambda y, t: f(y, t, torch.zeros_like(t))
        else:
            D = probs_fn(make_fn(teacher, teacher_cond, clock, vocab_size))
            Dbar = lambda y, t: D(y, t, torch.zeros_like(t))

        # diagonal (kappa = 0)
        t_d, w_d = sample_diag_time(nd, L.diag_time, L.t_eps, dev, rep, L.stratified)
        ctx = L.partial_context      # DBTM: some positions are clean context (time 1); the network is told no time
        x_d = sample_state(tok_diag, vocab_size, partial_context(t_d, tok_diag.shape[1]) if ctx and nd else t_d)

        # off-diagonal (kappa > 0)
        t_o, k3, k1 = sample_jumps(no, L.jump_sampling, clock, dev, L.off_diagonal == "semigroup", rep,
                                   L.whole_jump_prob, L.stratified, L.end_jump_prob)
        near = torch.zeros(no, dtype=torch.bool, device=dev)
        if L.boundary_prob > 0:
            near = torch.rand(no, device=dev) < L.boundary_prob
            t_o = torch.where(near, L.boundary_t + (1 - L.boundary_t) * torch.rand(no, device=dev), t_o)
            u = torch.rand(no, device=dev)
            k3 = torch.where(near, torch.where(u < 0.5, torch.ones_like(u), 2 * u - 1), k3)
        x_o = sample_state(tok_off, vocab_size, partial_context(t_o, tok_off.shape[1]) if ctx else t_o)
        with torch.no_grad():
            far = ~near
            target = torch.empty_like(x_o)
            if L.whole_jump_target == "semigroup" and L.off_diagonal != "semigroup":
                # The Lagrangian and Eulerian targets do not pin down the whole jump (at kappa = 1 they reduce to
                # D(delta) = delta and to a zero derivative, which a constant output satisfies): use the semigroup
                # target, which composes two shorter jumps, for those samples.
                whole = far & (k3 >= 1 - 1e-6)
                if whole.any():
                    kk1 = k1[whole] if k1 is not None else torch.rand(int(whole.sum()), device=dev) * 0.98 + 0.01
                    target[whole] = TARGETS["semigroup"](f, Dbar, x_o[whole], t_o[whole], k3[whole], kk1)
                    far = far & ~whole
            if far.any():
                Dfar = Dbar
                if L.direction == "data":   # the data point in place of the denoiser: a one-sample estimate
                    x1_far = torch.nn.functional.one_hot(tok_off[far], vocab_size).float()
                    Dfar = lambda y, t: x1_far
                target[far] = target_fn(f, Dfar, x_o[far], t_o[far], k3[far], None if k1 is None else k1[far],
                                        clock=clock, jvp=jvp)
            if near.any():
                if L.boundary_target == "data":
                    target[near] = torch.nn.functional.one_hot(tok_off[near], vocab_size).float()
                else:
                    target[near] = Dbar(x_o[near], t_o[near])
            if teacher is not None and use_diag:
                target_d = Dbar(x_d, t_d)
        tnet.train(was_training)

        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        logits = fn(torch.cat([x_d, x_o]), torch.cat([t_d, t_o]), torch.cat([torch.zeros_like(t_d), k3]))
        if not use_diag:
            loss_diag = logits.new_zeros(())
        elif teacher is None:
            loss_diag = diagonal_data_loss(logits[:nd], tok_diag, w_d)
        else:
            loss_diag = soft_cross_entropy(logits[:nd], target_d)
        loss_off = soft_cross_entropy(logits[nd:], target)

        loss = L.diag_weight * loss_diag + L.alpha * loss_off
        logs = {"diag": loss_diag.item(), "off": loss_off.item()}
        step = getattr(loss_fn, "step", 0)
        if L.ril_frac > 0 and step >= L.ril_start:       # refinement in the loop (DBTM), on the whole jump kappa = 1
            distill = L.ril_distill_weight > 0 and step >= max(L.ril_start, L.ril_distill_start)
            if distill and (getattr(loss_fn, "snapshot", None) is None or step - loss_fn.snapshot_step >= L.ril_refresh):
                loss_fn.snapshot = copy.deepcopy(raw).eval().requires_grad_(False)       # frozen teacher, refreshed
                loss_fn.snapshot_step = step
            n = max(2, int(L.ril_frac * B))
            x0 = torch.randn(B, tokens.shape[1], vocab_size, device=dev)
            if L.ril_ot:                                  # pairing over the whole batch, then a part of it is rolled out
                x0 = ot_coupling(x0, tokens)
            tok_r, x0 = tokens[:n], x0[:n]
            ones, zeros = torch.ones(n, device=dev), torch.zeros(n, device=dev)
            whole = lambda y: fn(y, zeros, ones)
            states, masks = refinement_rollout(whole, tok_r, x0, vocab_size, L.ril_depth, L.ril_threshold, L.ril_commit)
            target = None
            if distill and L.ril_distill_target == "sample":
                # the tokens the rollout itself ends with: the multi-round sample from this noise, as a hard target
                target = torch.nn.functional.one_hot(refinement_rollout.final_tokens, vocab_size).float()
            elif distill:
                snap = make_fn(loss_fn.snapshot, cfg.model.cond, clock, vocab_size)
                with torch.no_grad():
                    target = (snap(states[-1], zeros, ones).float() / L.ril_distill_temp).softmax(-1)
            adaptive = (L.ril_adaptive_c, L.ril_adaptive_r) if L.ril_adaptive_r > 0 else None
            ril_c = semi_r = 0.0
            for x_r, m_r in zip(states, masks):           # one round at a time, so that only one graph is in memory
                out = whole(x_r)
                c_r = refinement_round_loss(out, tok_r, m_r, None, adaptive) if L.ril_weight > 0 else out.new_zeros(())
                d_r = refinement_round_loss(out, tok_r, m_r, target, adaptive) if distill else out.new_zeros(())
                ((L.ril_weight * c_r + L.ril_distill_weight * d_r) / L.ril_depth).backward()
                ril_c, semi_r = ril_c + c_r.item() / L.ril_depth, semi_r + d_r.item() / L.ril_depth
            logs.update(ril=ril_c, ril_distill=semi_r)
        return loss, {"loss": loss.item(), **logs}

    return loss_fn
