"""The two training objectives, as functions  loss_fn(model, raw, tokens) -> (loss, logs).

`model` is the (possibly DDP-wrapped) network used for every forward pass that receives a gradient.
`raw` is the same network unwrapped; it is used to build targets, without gradient.
"""
import torch

import copy

from .losses import (TARGETS, lagrangian_target_positions, position_speed_jump, position_speed_state, confidence, adaptive_speed_state, adaptive_state, diagonal_data_loss, jvp_forward, make_jvp_fd, ot_coupling, partial_context, random_speed_times,
                     refinement_rollout, refinement_round_loss, sample_diag_time, sample_jumps, sample_state,
                     soft_cross_entropy, speed_clock_state, uniform01)
from .nets import make_fn, probs_fn
from .reparam import get_reparam


def _reparam(cfg, vocab_size):
    L = cfg.loss
    if cfg.model.cond != "duration" or L.diag_time == "reparam" or "reparam" in L.jump_sampling:
        return get_reparam(vocab_size)
    return None


def _adaptive(L, score_fn, tokens, vocab_size, t):
    """States of Section 7.2 for the times t: (x, clean)."""
    use = None if L.adaptive_frac >= 1 else torch.rand(tokens.shape[0], device=tokens.device) < L.adaptive_frac
    if L.adaptive == "speed":
        return adaptive_speed_state(score_fn, tokens, vocab_size, t, L.adaptive_beta, L.adaptive_score, L.adaptive_rollout_steps, use)
    steps = L.adaptive_rollout_steps if L.adaptive == "rollout" else 0
    other = None             # with `partial_context` too: the samples without model-chosen positions get random ones
    if L.partial_context and use is not None:
        other = partial_context(t, tokens.shape[1], L.context_frac) >= 1
    if L.adaptive == "speed_clock":
        return speed_clock_state(score_fn, tokens, vocab_size, t, get_reparam(vocab_size), L.adaptive_beta, L.adaptive_power,
                                 L.adaptive_base, L.adaptive_score, L.adaptive_rollout_steps, use, other)
    return adaptive_state(score_fn, tokens, vocab_size, t, L.adaptive_h, L.adaptive_score, steps, use, other)


def make_field_loss(cfg, vocab_size, clock):
    """Denoiser only (kappa = 0), trained with the weighted cross-entropy against the data."""
    L = cfg.loss
    rep = _reparam(cfg, vocab_size)

    def loss_fn(model, raw, tokens):
        t, w = sample_diag_time(tokens.shape[0], L.diag_time, L.t_eps, tokens.device, rep, L.stratified)
        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        noisy, logs = None, {}
        if L.adaptive != "off" and getattr(loss_fn, "step", 0) >= L.adaptive_start:
            # a denoiser has no map to the end: its own output scores the positions
            score = probs_fn(make_fn(raw, cfg.model.cond, clock, vocab_size))
            x, clean = _adaptive(L, lambda y, s: score(y, s, torch.zeros_like(s)), tokens, vocab_size, t)
            noisy, logs = ~clean, {"clean": clean.float().mean().item()}
        elif L.random_speed > 0:      # one time per position, as an adaptive-speed schedule with random confidences
            t_pos = random_speed_times(t, tokens.shape[1], L.random_speed, L.context_frac)
            x, noisy = sample_state(tokens, vocab_size, t_pos), t_pos < 1
            logs = {"clean": (t_pos >= 1).float().mean().item()}
        elif L.partial_context:
            t_pos = partial_context(t, tokens.shape[1], L.context_frac)
            x, noisy = sample_state(tokens, vocab_size, t_pos), (t_pos < 1) if L.freeze_context else None
        else:
            x = sample_state(tokens, vocab_size, t)
        loss = diagonal_data_loss(fn(x, t, torch.zeros_like(t)), tokens, w, noisy)
        return loss, {"loss": loss.item(), **logs}

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
    sc = cfg.model.self_cond
    if sc:
        assert L.ril_frac == 0 and L.adaptive == "off" and L.position_speed == 0, "self-conditioning: plain states only"

    tnet, was_training = None, False

    def position_speed_loss(model, f, Dbar, tok_diag, tok_off):
        """Flow map with one duration per position, distilled from a denoiser: delta_hat(x, kappa), kappa in [0, 1]^L,
        Phi^l = (1 - kappa^l) x^l + kappa^l delta_hat^l. States have one progress per position; the jump is one
        adaptive-speed step, with the teacher's confidences. Semigroup target: first leg a kappa, a ~ U[0, 1]."""
        assert teacher is not None and L.boundary_prob == 0 and clock.name == "exponential" and cfg.model.cond == "duration"
        dev, nd, no = tok_off.device, tok_diag.shape[0], tok_off.shape[0]
        t_d, _ = sample_diag_time(nd, L.diag_time, L.t_eps, dev, rep, L.stratified)
        t_o, k3, k1 = sample_jumps(no, L.jump_sampling, clock, dev, True, rep, L.whole_jump_prob, L.stratified, L.end_jump_prob)
        a = (k1 / k3.clamp_min(1e-8)).clamp(0.01, 0.99).view(-1, 1)
        # the durations of the notes mostly land where the decoding clock has run out; on a part of the samples the jump
        # is a step inside the clock instead, as the sampler makes: tau(t') ~ U[tau(t), 1]
        short = (torch.rand(no, device=dev) < L.position_speed_short) & (k3 < 1 - 1e-6)
        s_o = rep.tau(t_o)
        t_short = rep.t(s_o + torch.rand(no, device=dev) * (1 - s_o))
        k3 = torch.where(short, ((t_short - t_o) / (1 - t_o).clamp_min(1e-8)).clamp(0, 1), k3)
        x_d, u_d, _ = position_speed_state(tok_diag, vocab_size, t_d, rep, L.position_speed)
        x_o, u_o, beta = position_speed_state(tok_off, vocab_size, t_o, rep, L.position_speed)
        with torch.no_grad():
            g = confidence(Dbar(x_o, t_o), "maxprob")
            k = position_speed_jump(g, u_o, t_o, k3, rep, beta, L.position_speed_power)
            if L.off_diagonal == "semigroup":
                target = TARGETS["semigroup"](f, Dbar, x_o, t_o.view(-1, 1), k, a * k)
                fin = (k >= 1 - 1e-6) & (k3 < 1 - 1e-6).view(-1, 1)      # positions that finish inside a partial jump
                if L.position_speed_finish == "teacher" and fin.any():
                    # their target is the teacher's prediction given where the other positions land (as in
                    # lagrangian_target_positions), not the model's own second leg
                    km = torch.where(k >= 1 - 1e-6, torch.zeros_like(k), k).unsqueeze(-1)
                    x_land = (1 - km) * x_o + km * f(x_o, t_o, k)
                    target = torch.where(fin.unsqueeze(-1), Dbar(x_land, t_o), target)
            else:
                target = lagrangian_target_positions(f, Dbar, x_o, k, jvp)
            target_d = Dbar(x_d, t_d)
        tnet.train(was_training)
        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        t_all = torch.cat([t_d, t_o])
        logits = fn(torch.cat([x_d, x_o]), t_all, torch.cat([torch.zeros_like(u_d), k]))
        loss_diag = soft_cross_entropy(logits[:nd], target_d, u_d < 1)
        loss_off = soft_cross_entropy(logits[nd:], target, u_o < 1)
        loss = L.diag_weight * loss_diag + L.alpha * loss_off
        return loss, {"loss": loss.item(), "diag": loss_diag.item(), "off": loss_off.item(),
                      "clean": (u_o >= 1).float().mean().item(), "kappa": k[u_o < 1].mean().item()}

    def loss_fn(model, raw, tokens):
        nonlocal tnet, was_training
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

        if L.position_speed > 0:
            return position_speed_loss(model, f, Dbar, tok_diag, tok_off)

        def bind(draft):
            """(f, Dbar) told the draft `draft` (self-conditioning); a teacher is told no draft."""
            fb = probs_fn(make_fn(tnet, cfg.model.cond, clock, vocab_size, sc=draft,
                                  **({"manual_attn": True} if need_jvp and forward_mode else {})))
            return fb, (Dbar if teacher is not None else lambda y, t: fb(y, t, torch.zeros_like(t)))

        def draft(tok, x, t):
            """Self-conditioning input: the denoiser output without draft, from the same state or from an independent
            noisy view of the same sequence; zero (no draft) on 1 - self_cond_prob of the rows."""
            n = tok.shape[0]
            if n == 0:
                return x.new_zeros(x.shape)
            if L.self_cond_view == "other":
                t, _ = sample_diag_time(n, L.diag_time, L.t_eps, dev, rep, L.stratified)
                x = sample_state(tok, vocab_size, partial_context(t, tok.shape[1], L.context_frac) if ctx else t)
            with torch.no_grad():
                p = f(x, t, torch.zeros_like(t))
            return p * (torch.rand(n, device=dev) < L.self_cond_prob).view(-1, 1, 1).to(p.dtype)

        # diagonal (kappa = 0)
        t_d, w_d = sample_diag_time(nd, L.diag_time, L.t_eps, dev, rep, L.stratified)
        ctx = L.partial_context      # DBTM: some positions are clean context (time 1); the network is told no time
        adaptive = L.adaptive != "off" and getattr(loss_fn, "step", 0) >= L.adaptive_start
        to_end = lambda y, s: f(y, s, torch.ones_like(s))         # the map to the end scores the positions
        clean_d = clean_o = None
        if adaptive and nd:
            x_d, clean_d = _adaptive(L, to_end, tok_diag, vocab_size, t_d)
        elif ctx and nd:
            tp = partial_context(t_d, tok_diag.shape[1], L.context_frac)
            x_d, clean_d = sample_state(tok_diag, vocab_size, tp), tp >= 1
        else:
            x_d = sample_state(tok_diag, vocab_size, t_d)

        # off-diagonal (kappa > 0)
        t_o, k3, k1 = sample_jumps(no, L.jump_sampling, clock, dev, L.off_diagonal == "semigroup", rep,
                                   L.whole_jump_prob, L.stratified, L.end_jump_prob)
        near = torch.zeros(no, dtype=torch.bool, device=dev)
        if L.boundary_prob > 0:
            near = torch.rand(no, device=dev) < L.boundary_prob
            t_o = torch.where(near, L.boundary_t + (1 - L.boundary_t) * torch.rand(no, device=dev), t_o)
            u = torch.rand(no, device=dev)
            k3 = torch.where(near, torch.where(u < 0.5, torch.ones_like(u), 2 * u - 1), k3)
        if adaptive:
            x_o, clean_o = _adaptive(L, to_end, tok_off, vocab_size, t_o)
        elif ctx:
            tp = partial_context(t_o, tok_off.shape[1], L.context_frac)
            x_o, clean_o = sample_state(tok_off, vocab_size, tp), tp >= 1
        else:
            x_o = sample_state(tok_off, vocab_size, t_o)
        if not (L.freeze_context or adaptive):
            clean_d = clean_o = None
        fz = lambda m: {} if clean_o is None else {"frozen": clean_o[m]}
        p_d = draft(tok_diag, x_d, t_d) if sc else None
        p_o = draft(tok_off, x_o, t_o) if sc else None
        bound = lambda m: bind(p_o[m]) if sc else (f, Dbar)      # the targets are built under the row's own draft
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
                    target[whole] = TARGETS["semigroup"](*bound(whole), x_o[whole], t_o[whole], k3[whole], kk1, **fz(whole))
                    far = far & ~whole
            if far.any():
                f_far, Dfar = bound(far)
                if L.direction == "data":   # the data point in place of the denoiser: a one-sample estimate
                    x1_far = torch.nn.functional.one_hot(tok_off[far], vocab_size).float()
                    Dfar = lambda y, t: x1_far
                target[far] = target_fn(f_far, Dfar, x_o[far], t_o[far], k3[far], None if k1 is None else k1[far],
                                        clock=clock, jvp=jvp, **fz(far))
            if near.any():
                if L.boundary_target == "data":
                    target[near] = torch.nn.functional.one_hot(tok_off[near], vocab_size).float()
                else:
                    target[near] = bound(near)[1](x_o[near], t_o[near])
            if teacher is not None and use_diag:
                target_d = Dbar(x_d, t_d)
        tnet.train(was_training)

        fn = make_fn(model, cfg.model.cond, clock, vocab_size, **({"sc": torch.cat([p_d, p_o])} if sc else {}))
        logits = fn(torch.cat([x_d, x_o]), torch.cat([t_d, t_o]), torch.cat([torch.zeros_like(t_d), k3]))
        if not use_diag:
            loss_diag = logits.new_zeros(())
        elif teacher is None:
            loss_diag = diagonal_data_loss(logits[:nd], tok_diag, w_d, None if clean_d is None else ~clean_d)
        else:
            loss_diag = soft_cross_entropy(logits[:nd], target_d, None if clean_d is None else ~clean_d)
        loss_off = soft_cross_entropy(logits[nd:], target, None if clean_o is None else ~clean_o)

        loss = L.diag_weight * loss_diag + L.alpha * loss_off
        logs = {"diag": loss_diag.item(), "off": loss_off.item()}
        if clean_o is not None:
            logs["clean"] = clean_o.float().mean().item()
        step = getattr(loss_fn, "step", 0)
        if L.ril_frac > 0 and step >= L.ril_start:       # refinement in the loop (DBTM), on the whole jump kappa = 1
            distill = L.ril_distill_weight > 0 and step >= max(L.ril_start, L.ril_distill_start)
            frozen = L.ril_rollout_net == "snapshot"      # the frozen copy plays the rollout, not the trained network
            if (distill or frozen) and (getattr(loss_fn, "snapshot", None) is None or step - loss_fn.snapshot_step >= L.ril_refresh):
                loss_fn.snapshot = copy.deepcopy(raw).eval().requires_grad_(False)       # frozen teacher, refreshed
                loss_fn.snapshot_step = step
            n = max(2, int(L.ril_frac * B))
            x0 = torch.randn(B, tokens.shape[1], vocab_size, device=dev)
            if L.ril_ot:                                  # pairing over the whole batch, then a part of it is rolled out
                x0 = ot_coupling(x0, tokens)
            tok_r, x0 = tokens[:n], x0[:n]
            ones, zeros = torch.ones(n, device=dev), torch.zeros(n, device=dev)
            whole = lambda y: fn(y, zeros, ones)
            roll = whole
            if frozen:
                snap_roll = make_fn(loss_fn.snapshot, cfg.model.cond, clock, vocab_size)
                roll = lambda y: snap_roll(y, zeros, ones)
            states, masks = refinement_rollout(roll, tok_r, x0, vocab_size, L.ril_depth, L.ril_threshold, L.ril_commit)
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


def make_val_loss(cfg, vocab_size, clock):
    """Held-out denoising loss: returns val(model) -> mean cross-entropy (nats per token) of the denoiser (kappa = 0)
    on `eval.val_num` sequences of `eval.val_path`, at times spread uniformly in the decoding-error clock, with fixed
    noise. The same number for every model that is told the same things, so curves and runs can be compared."""
    import numpy as np
    if not cfg.eval.val_path:
        return None
    data = np.load(cfg.eval.val_path, mmap_mode="r")
    tokens = torch.from_numpy(np.asarray(data[:cfg.eval.val_num]).astype(np.int64))
    rep = get_reparam(vocab_size)
    n = tokens.shape[0]
    times = rep.t((torch.arange(n, dtype=torch.float64) + 0.5) / n).float()[torch.randperm(n, generator=torch.Generator().manual_seed(0))]

    @torch.no_grad()
    def val(model):
        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()
        fn = make_fn(model, cfg.model.cond, clock, vocab_size)
        gen = torch.Generator(device=dev).manual_seed(0)
        total = 0.0
        for i in range(0, n, 128):
            tok, t = tokens[i:i + 128].to(dev), times[i:i + 128].to(dev)
            x1 = torch.nn.functional.one_hot(tok, vocab_size).float()
            x = (1 - t.view(-1, 1, 1)) * torch.randn(x1.shape, device=dev, generator=gen) + t.view(-1, 1, 1) * x1
            logits = fn(x, t, torch.zeros_like(t))
            total += torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), tok.flatten(), reduction="sum").item()
        model.train(was_training)
        return total / tokens.numel()

    return val
