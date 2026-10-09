"""Discrete Beckmann Transport Models (Tang and Wang, arXiv 2609.15903), as a baseline, following their Algorithms 1 to 4.

One time-free network T(x) (here: the DiT called with duration 1) maps a state straight to the data. Training, per batch:

    pairing     noise x_0 and data x_1 are paired by minibatch optimal transport            (dbtm.ot)
    context     per sample, a fraction f ~ U[0, 1] of the positions is clean, t^l = 1        (dbtm.partial_context)
    state       I^l = (1 - t^l) x_0^l + t^l x_1^l with t ~ U[0, 1]; velocity dI^l = x_1^l - x_0^l on noisy positions
    transport   | T(I) - sg( T(I) + dI . grad T(I) ) |^2 on the noisy positions              (the conservation equation)
    anchor      cross-entropy of T(I) to x_1 on the positions with t^l > dbtm.t_anchor
    boundary    cross-entropy of T(x_1) to x_1
    refinement  from step dbtm.ril_start: the rollout losses of deqfm/losses.py (data-token commits, ril-c and semi-r)

Each sequence's transport, anchor and refinement loss is weighted by (loss + c)^(-r), no gradient through the weight
("adaptive" losses of the paper, r = 0.5, c = 1). The paper does not give the weights of the rollout losses; they are 1.
"""
import copy

import torch
import torch.nn.functional as F

from .losses import ot_coupling, partial_context, refinement_rollout, refinement_round_loss
from .track import nb


def _adaptive(per_sequence, c, r):
    return (per_sequence * (per_sequence.detach() + c) ** (-r)).mean() if r > 0 else per_sequence.mean()


@nb.core
def dbtm_base_losses(logits_fn, tokens, x0, t_pos, vocab_size, t_anchor, adaptive, jvp="forward", fd_eps=1e-3):
    """Transport and anchor losses on the states I = (1 - t) x_0 + t x_1 with per-position times t_pos [B, L].
    logits_fn(x) -> logits of T(x). Returns (transport, anchor)."""
    x1 = F.one_hot(tokens, vocab_size).float()
    tl = t_pos.unsqueeze(-1)
    state = (1 - tl) * x0 + tl * x1
    noisy = t_pos < 1
    velocity = (x1 - x0) * noisy.unsqueeze(-1)                       # clean positions do not move
    probs = lambda y: logits_fn(y).float().softmax(-1)
    with torch.no_grad():                                            # target: T + dI . grad T, fully stopped
        if jvp == "forward":
            out, deriv = torch.func.jvp(probs, (state,), (velocity,))
        else:
            out = probs(state)
            deriv = (probs(state + fd_eps * velocity) - probs(state - fd_eps * velocity)) / (2 * fd_eps)
        target = out + deriv
    logits = logits_fn(state)
    sq = (logits.float().softmax(-1) - target).pow(2).sum(-1)       # [B, L]
    transport = _adaptive((sq * noisy).sum(-1) / noisy.sum(-1).clamp_min(1), *adaptive)
    ce = F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten(), reduction="none").view(tokens.shape)
    late = t_pos > t_anchor
    anchor = _adaptive((ce * late).sum(-1) / late.sum(-1).clamp_min(1), *adaptive)
    return transport, anchor


def make_dbtm_loss(cfg, vocab_size):
    D = cfg.dbtm
    adaptive = (D.adaptive_c, D.adaptive_r)

    def loss_fn(model, raw, tokens):
        B, L = tokens.shape
        dev = tokens.device
        ones = torch.ones(B, device=dev)
        T = lambda y: model(y, ones[:y.shape[0]], manual_attn=True)
        x0 = torch.randn(B, L, vocab_size, device=dev)
        if D.ot:
            x0 = ot_coupling(x0, tokens)
        t = torch.rand(B, device=dev)
        t_pos = partial_context(t, L) if D.partial_context else t.view(-1, 1).expand(-1, L)
        transport, anchor = dbtm_base_losses(T, tokens, x0, t_pos, vocab_size, D.t_anchor, adaptive, cfg.loss.jvp, cfg.loss.fd_eps)
        x1 = F.one_hot(tokens, vocab_size).float()
        boundary = F.cross_entropy(T(x1).float().flatten(0, 1), tokens.flatten())
        loss = transport + D.lambda_anchor * anchor + D.lambda_bnd * boundary
        logs = {"transport": transport.item(), "anchor": anchor.item(), "boundary": boundary.item()}

        step = getattr(loss_fn, "step", 0)
        if D.ril and step >= D.ril_start:
            if getattr(loss_fn, "snapshot", None) is None or step - loss_fn.snapshot_step >= D.ril_refresh:
                loss_fn.snapshot = copy.deepcopy(raw).eval().requires_grad_(False)       # frozen teacher, refreshed
                loss_fn.snapshot_step = step
            n = max(2, int(D.ril_frac * B))
            tok_r, x0_r, ones_r = tokens[:n], x0[:n], ones[:n]
            whole = lambda y: model(y, ones_r)
            states, masks = refinement_rollout(whole, tok_r, x0_r, vocab_size, D.ril_depth, D.ril_threshold, "data")
            with torch.no_grad():
                target = (loss_fn.snapshot(states[-1], ones_r).float() / D.ril_distill_temp).softmax(-1)
            ril_c = semi_r = 0.0
            for x_r, m_r in zip(states, masks):                   # one round at a time: one graph in memory
                out = whole(x_r)
                c_r = refinement_round_loss(out, tok_r, m_r, None, adaptive)
                d_r = refinement_round_loss(out, tok_r, m_r, target, adaptive)
                ((D.lambda_ril * c_r + D.lambda_semi * d_r) / D.ril_depth).backward()
                ril_c, semi_r = ril_c + c_r.item() / D.ril_depth, semi_r + d_r.item() / D.ril_depth
            logs.update(ril=ril_c, ril_distill=semi_r)
        return loss, {"loss": loss.item(), **logs}

    return loss_fn
