"""Losses and off-diagonal targets (Algorithms 1 to 4 of the notes).

Notation
    x           state in R^{B x L x V}, at nominal time t in [0, 1]^B
    f(x, t, k)  model probabilities for a jump that covers the fraction k = kappa of the way to the denoised point
    Dbar(x, t)  teacher denoiser probabilities: f(x, t, 0) when training from data, a frozen model when distilling
    clock       gives c(sigma), kappa(sigma) and kappa^{-1}; the network sees sigma = kappa^{-1}(k)

A jump of fraction k from time t lands at time 1 - (1 - k)(1 - t). Networks that are time-free ignore t.

Every target has rows that sum to one, and no gradient flows through a target.
All losses are averaged over positions and over the batch.
"""
import math

import torch
import torch.nn.functional as F


from .track import nb

def _b(v):
    """[B] -> [B, 1, 1] (or [B, L] -> [B, L, 1], one time per position) for broadcasting against states."""
    return v.unsqueeze(-1) if v.dim() == 2 else v.view(-1, 1, 1)


# ----------------------------------------------------------------------------- states and times

@nb.core
def sample_state(tokens, vocab_size, t):
    """I_t = (1 - t) x_0 + t x_1 with x_0 ~ N(0, I) and x_1 the one-hot matrix of `tokens`. t: [B], or [B, L]."""
    x1 = F.one_hot(tokens, vocab_size).float()
    return (1 - _b(t)) * torch.randn_like(x1) + _b(t) * x1


@nb.core
def partial_context(t, seq_len, frac=1.0):
    """Partial-context interpolant of Discrete Beckmann Transport Models: per sample, a fraction f ~ U[0, 1] of the
    positions (each with probability f) is clean context, t = 1; the others keep the sample's time. Returns [B, L].
    `frac` < 1: only that fraction of the samples gets clean positions."""
    f = torch.rand(t.shape[0], 1, device=t.device)
    clean = torch.rand(t.shape[0], seq_len, device=t.device) < f
    if frac < 1:
        clean = clean & (torch.rand(t.shape[0], 1, device=t.device) < frac)
    return torch.where(clean, torch.ones_like(f), t.view(-1, 1).expand(-1, seq_len))


@nb.core
def confidence(p, kind="maxprob"):
    """Confidence score g^l in [0, 1] of each position, from probabilities p [B, L, V] (equation 33 of the notes).
    maxprob: largest probability. margin: largest minus second largest. certainty: 1 - entropy / log V."""
    if kind == "maxprob":
        return p.max(-1).values
    if kind == "margin":
        top = p.topk(2, dim=-1).values
        return top[..., 0] - top[..., 1]
    if kind == "certainty":
        return 1 + (p * p.clamp_min(1e-30).log()).sum(-1) / math.log(p.shape[-1])
    raise ValueError(f"unknown confidence score '{kind}'")


@nb.core
@torch.no_grad()
def adaptive_state(score_fn, tokens, vocab_size, t, h, kind="maxprob", rollout_steps=0, use=None, other=None):
    """State with clean positions chosen by the model (Section 7.2 of the notes, binary version).

    score_fn(x) -> probabilities of the model at the state x (no gradient). Returns (x, clean):
        I_t = (1 - t) x_0 + t x_1,  p = score_fn(I_t),  C = {l : g^l > h},  x^l = x_1^l on C and I_t^l elsewhere.
    rollout_steps = n > 0: C is built step by step, as the sampler does. For j = 1..n, at time t_j = j t / n, the
    scores are taken at the state with the positions committed so far already clean, and C grows.
    `use` [B] bool: samples that get adaptive states; the others keep the plain I_t."""
    x1 = F.one_hot(tokens, vocab_size).float()
    x0 = torch.randn_like(x1)
    clean = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
    for j in range(1, max(rollout_steps, 1) + 1):
        tj = t * j / max(rollout_steps, 1)
        x = torch.where(clean.unsqueeze(-1), x1, (1 - _b(tj)) * x0 + _b(tj) * x1)
        clean = clean | (confidence(score_fn(x, tj), kind) > h)
    if use is not None:      # the other samples: plain, or random clean positions when `other` [B, L] is given
        clean = torch.where(use.view(-1, 1), clean, torch.zeros_like(clean) if other is None else other)
    return torch.where(clean.unsqueeze(-1), x1, (1 - _b(t)) * x0 + _b(t) * x1), clean


@nb.core
@torch.no_grad()
def adaptive_speed_state(score_fn, tokens, vocab_size, t, beta, kind="maxprob", steps=8, use=None):
    """State with one time per position (Section 7.2 of the notes, continuous version, equations 31 and 32).

    t^l(0) = 0 and dt^l / dt = psi(g^l) with psi(g) = 1 + beta g, g^l the confidence of the model at the current
    state; t^l stops at 1. Integrated with `steps` Euler steps from 0 to t, the model in the loop (no gradient):
        x_j^l = (1 - t_j^l) x_0^l + t_j^l x_1^l,   g_j = confidence(score_fn(x_j)),   t_{j+1}^l = min(1, t_j^l + (t / steps) psi(g_j^l)).
    Returns (x, clean): the state at the final times and the positions that reached t^l = 1 (the data token)."""
    x1 = F.one_hot(tokens, vocab_size).float()
    x0 = torch.randn_like(x1)
    tl = torch.zeros(tokens.shape, device=tokens.device)
    for j in range(steps):
        x = (1 - tl.unsqueeze(-1)) * x0 + tl.unsqueeze(-1) * x1
        g = confidence(score_fn(x, t * j / steps), kind)
        tl = (tl + (t / steps).view(-1, 1) * (1 + beta * g)).clamp(max=1.0)
    if use is not None:
        tl = torch.where(use.view(-1, 1), tl, t.view(-1, 1).expand_as(tl))
    return (1 - tl.unsqueeze(-1)) * x0 + tl.unsqueeze(-1) * x1, tl >= 1


@nb.core
def random_speed_times(t, seq_len, beta_max, frac=1.0):
    """Per-position times of an adaptive-speed schedule with random confidences (no model in the loop):
    t^l = min(1, t (1 + beta g^l)), g^l ~ U[0, 1] independently per position, beta ~ U[0, beta_max] per sample.
    This is where the speed 1 + beta g^l takes position l by time t when g^l stays constant. Returns [B, L].
    `frac` < 1: only that fraction of the samples; the others keep the common time t."""
    beta = beta_max * torch.rand(t.shape[0], 1, device=t.device)
    if frac < 1:
        beta = beta * (torch.rand(t.shape[0], 1, device=t.device) < frac)
    return (t.view(-1, 1) * (1 + beta * torch.rand(t.shape[0], seq_len, device=t.device))).clamp(max=1.0)


@nb.core
@torch.no_grad()
def speed_clock_state(score_fn, tokens, vocab_size, t, reparam, beta, power, base=1.0, kind="maxprob", steps=8, use=None,
                      other=None):
    """State of the continuous adaptive-speed process in the decoding-error clock (the training side of
    `sampling.sample_speed_clock`), with the model in the loop and no gradient.

    s = tau(t) is the global progress of the sample. Every position has its own progress u^l, from u^l = 0:
        x_j^l = (1 - t_j^l) x_0^l + t_j^l x_1^l with t_j^l = tau^{-1}(u_j^l),   g_j = confidence(score_fn(x_j)),
        u_{j+1}^l = min(1, u_j^l + (s / n) (base + beta (g_j^l)^power)),        j = 0..n-1.
    Returns (x_n, clean), clean = positions with u^l = 1 (the data token). `use` [B]: samples that get these states;
    the others get the common time t, with the clean positions `other` [B, L] if given."""
    x1 = F.one_hot(tokens, vocab_size).float()
    x0 = torch.randn_like(x1)
    s = reparam.tau(t)
    u = torch.zeros(tokens.shape, device=tokens.device)

    def state(u):
        tl = torch.where(u >= 1, torch.ones_like(u), reparam.t(u)).unsqueeze(-1)
        return (1 - tl) * x0 + tl * x1

    for j in range(steps):
        g = confidence(score_fn(state(u), t), kind)
        u = (u + (s / steps).view(-1, 1) * (base + beta * g.pow(power))).clamp(max=1.0)
    if use is not None:
        plain = s.view(-1, 1).expand_as(u)
        if other is not None:
            plain = torch.where(other, torch.ones_like(plain), plain)
        u = torch.where(use.view(-1, 1), u, plain)
    return state(u), u >= 1


def uniform01(batch, device, stratified=False):
    """U[0, 1] samples; `stratified` puts one sample in each of `batch` equal bins, in random order (as FLM does)."""
    u = torch.rand(batch, device=device)
    if stratified:
        u = ((u + torch.arange(batch, device=device)) / batch)[torch.randperm(batch, device=device)]
    return u


@nb.core
def sample_diag_time(batch, mode, t_eps, device, reparam=None, stratified=False):
    """Time t ~ q and loss weight for the diagonal loss.

    uniform     : q = U[0, 1 - t_eps], weight 1 / (1 - t)   (the form in the notes, with a numerical guard)
    log_uniform : q(t) prop. to 1 / (1 - t) on [0, 1 - t_eps], so the weight is the constant log(1 / t_eps)
    reparam     : tau(t) ~ U[0, 1] with the decoding-error clock of FLM, weight 1
    """
    u = uniform01(batch, device, stratified)
    if mode == "reparam":
        return reparam.t(u), torch.ones_like(u)
    if mode == "uniform":
        t = u * (1 - t_eps)
        return t, 1 / (1 - t)
    if mode == "log_uniform":
        t = 1 - t_eps ** u
        return t, torch.full_like(t, math.log(1 / t_eps))
    raise ValueError(f"unknown diag_time '{mode}'")


# ----------------------------------------------------------------------------- losses

@nb.core
def soft_cross_entropy(logits, target, noisy=None):
    """- target . log softmax(logits), averaged over positions and batch.

    When the rows of `target` sum to one, the gradient with respect to the logits is softmax(logits) - target.
    `noisy` [B, L] restricts the sum to the positions that are not clean context.
    """
    ce = -(target.float() * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    return ce.mean() if noisy is None else (ce * noisy).mean()


@nb.core
def diagonal_data_loss(logits, tokens, weight, noisy=None):
    """Weighted cross-entropy against the data: mean over the batch of weight * (- x_1 . log delta_hat_0(I_t)) / L.
    `noisy` [B, L] restricts the sum to the positions that are not clean context (equation 30 of the notes)."""
    ce = F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten(), reduction="none").view(tokens.shape)
    if noisy is not None:
        ce = ce * noisy
    return (weight * ce.mean(dim=1)).mean()


# ----------------------------------------------------------------------------- Jacobian-vector products

def jvp_forward(f, x, sigma, u, v):
    """(f(x, sigma), d/de f(x + e u, sigma + e v) at e = 0) by forward-mode differentiation."""
    return torch.func.jvp(f, (x, sigma), (u, v))


def make_jvp_fd(eps):
    def jvp_fd(f, x, sigma, u, v):
        """Same, by central finite differences (three evaluations; works with any attention kernel)."""
        out = f(x, sigma)
        plus, minus = f(x + eps * u, sigma + eps * v), f(x - eps * u, sigma - eps * v)
        return out, (plus - minus) / (2 * eps)
    return jvp_fd


@nb.core
def position_speed_state(tokens, vocab_size, t, reparam, beta_max):
    """State with one progress per position, for the flow map with one duration per position (no model in the loop).

    s = tau(t) is the common progress in the decoding-error clock. Per sample beta ~ U[0, beta_max]; per position
        u^l = min(1, s (1 + beta r^l)), r^l ~ U[0, 1],     t^l = tau^{-1}(u^l),     x^l = (1 - t^l) x_0^l + t^l x_1^l.
    Returns (x, u [B, L], beta [B, 1]); positions with u^l = 1 are clean (the data token). beta_max = 0: the common time t."""
    x1 = F.one_hot(tokens, vocab_size).float()
    beta = beta_max * torch.rand(t.shape[0], 1, device=t.device)
    u = (reparam.tau(t).view(-1, 1) * (1 + beta * torch.rand(tokens.shape, device=t.device))).clamp(max=1.0)
    tl = torch.where(u >= 1, torch.ones_like(u), reparam.t(u)).unsqueeze(-1)
    return (1 - tl) * torch.randn_like(x1) + tl * x1, u, beta


@nb.core
def position_speed_jump(g, u, t, k3, reparam, beta, power=1.0):
    """Jump fractions kappa^l [B, L] of one adaptive-speed step from the progress u, for confidences g [B, L].

    A plain jump of fraction k3 from the common time t lands at t' = 1 - (1 - k3)(1 - t), a step du = tau(t') - tau(t). Then
        u'^l = min(1, u^l + du (1 + beta (g^l)^power)),     kappa^l = (tau^{-1}(u'^l) - t^l) / (1 - t^l),
    so a position that reaches u'^l = 1 goes to the end (kappa^l = 1). When the plain jump itself lands where the clock
    has run out (tau(t') = 1, t' < 1), every position lands at t' instead. kappa^l = 0 at the clean positions and
    kappa^l = 1 everywhere when k3 = 1. beta = 0 and a common progress: kappa^l = k3."""
    tp = land(t, k3)
    tau_p = reparam.tau(tp)
    du = (tau_p - reparam.tau(t)).clamp_min(0.0).view(-1, 1)
    u2 = (u + du * (1 + beta * g.pow(power))).clamp(max=1.0)
    tl = torch.where(u >= 1, torch.ones_like(u), reparam.t(u))
    t2 = torch.where(u2 >= 1, torch.ones_like(u2), reparam.t(u2))
    t2 = torch.where((tau_p >= 1 - 1e-4).view(-1, 1), torch.maximum(tp.view(-1, 1), tl), t2)
    t2 = torch.where((k3 >= 1 - 1e-6).view(-1, 1), torch.ones_like(t2), t2)
    k = ((t2 - tl) / (1 - tl).clamp_min(1e-8)).clamp(0, 1)
    return torch.where(u >= 1, torch.zeros_like(k), k)


# ----------------------------------------------------------------------------- jump sampling

@nb.core
def sample_jumps(batch, mode, clock, device, need_mid, reparam=None, boundary_prob=0.0, stratified=False, end_prob=0.0):
    """State time t, jump fraction k3 and (for the semigroup loss) first-leg fraction k1, each of shape [batch].

    notes   : t ~ U[0, 1]. Semigroup: sigma_3 with density 2 sigma_3, sigma_1 ~ U[0, sigma_3]. Otherwise sigma ~ U[0, 1].
    notes_reparam: the same durations, with the state time uniform in the decoding-error clock, tau(t) ~ U[0, 1].
    endpoint: t ~ U[0, 1] and kappa = 1 always: the map to the end of the flow only (not for the semigroup loss).
    reparam : as FMLM, in the reparameterised time tau: h ~ U[0, 1], tau_s ~ U[0, 1 - h], tau_t = tau_s + h, midpoint
              (tau_s + tau_t) / 2; with probability `boundary_prob` the pair is (0, 1), a whole one-step jump;
              with probability `end_prob` it is (tau_s, 1) with tau_s ~ U[0, 1], a jump to the end from any time.
    """
    if mode in ("notes", "notes_reparam"):   # notes_reparam: the durations of the notes, the state time as FLM (tau uniform)
        t = uniform01(batch, device, stratified)
        if mode == "notes_reparam":
            t = reparam.t(t)
        whole = torch.rand(batch, device=device) < boundary_prob      # forced whole jump from noise, as FMLM (off by default)
        t = torch.where(whole, torch.zeros_like(t), t)
        if need_mid:
            s3 = torch.where(whole, torch.ones_like(t), torch.rand(batch, device=device).sqrt())
            s1 = torch.rand(batch, device=device) * s3
            return t, clock.kappa(s3), clock.kappa(s1)
        return t, clock.kappa(torch.where(whole, torch.ones_like(t), torch.rand(batch, device=device))), None
    if mode == "endpoint":       # only the whole jump to the end, from a state at a uniform time
        return uniform01(batch, device, stratified), torch.ones(batch, device=device), None
    if mode == "reparam":
        h = uniform01(batch, device, stratified)
        tau_s = torch.rand(batch, device=device) * (1 - h)
        if boundary_prob > 0:
            whole = torch.rand(batch, device=device) < boundary_prob
            tau_s, h = torch.where(whole, torch.zeros_like(h), tau_s), torch.where(whole, torch.ones_like(h), h)
        if end_prob > 0:      # jumps to the end from a state at any time: tau_s ~ U[0, 1], tau_t = 1
            end = torch.rand(batch, device=device) < end_prob
            tau_e = torch.rand(batch, device=device)
            tau_s, h = torch.where(end, tau_e, tau_s), torch.where(end, 1 - tau_e, h)
        tau_t = (tau_s + h).clamp(max=1.0)
        exact = tau_t >= 1.0 - 1e-6
        s, e, u = reparam.t(tau_s), reparam.t(tau_t), reparam.t(0.5 * (tau_s + tau_t))
        rem = (1 - s).clamp_min(1e-8)
        k3, k1 = ((e - s) / rem).clamp(0, 1), ((u - s) / rem).clamp(0, 1)
        k3 = torch.where(exact, torch.ones_like(k3), k3)      # a jump to the end is exactly kappa = 1
        return s, k3, torch.minimum(k1, k3)
    raise ValueError(f"unknown jump_sampling '{mode}'")


def land(t, k):
    """Nominal time after a jump of fraction k from time t."""
    return 1 - (1 - k) * (1 - t)


# ----------------------------------------------------------------------------- off-diagonal targets

@nb.core
@torch.no_grad()
def semigroup_target(f, Dbar, x, t, k3, k1, clock=None, jvp=None, frozen=None):
    """Algorithm 2: target = gamma f(x, k1) + (1 - gamma) f(x', k2), with x' the state after the first leg.
    `frozen` [B, L]: clean context positions, which a jump does not move."""
    k2 = ((k3 - k1) / (1 - k1).clamp_min(1e-8)).clamp(0, 1)
    gamma = k1 * (1 - k2) / k3.clamp_min(1e-8)
    d1 = f(x, t, k1)
    x_mid = (1 - _b(k1)) * x + _b(k1) * d1
    if frozen is not None:
        x_mid = torch.where(frozen.unsqueeze(-1), x, x_mid)
    d2 = f(x_mid, land(t, k1), k2)
    return _b(gamma) * d1 + (1 - _b(gamma)) * d2


def _in_sigma(f):
    """f as a function of (x, sigma), for derivatives with respect to the duration label."""
    if not hasattr(f, "in_sigma"):
        raise ValueError("the Lagrangian and Eulerian losses need model.cond = duration or state_time")
    return f.in_sigma


@nb.core
@torch.no_grad()
def lagrangian_target(f, Dbar, x, t, k3, k1=None, clock=None, jvp=jvp_forward, frozen=None):
    """Algorithm 3: target = Dbar(x') - c kappa d/dsigma f(x, sigma), with x' the state after the jump."""
    s = clock.kappa_inv(k3)
    d, J = jvp(_in_sigma(f), x, s, torch.zeros_like(x), torch.ones_like(s))
    x_end = (1 - _b(k3)) * x + _b(k3) * d
    if frozen is not None:
        x_end = torch.where(frozen.unsqueeze(-1), x, x_end)
    return Dbar(x_end, land(t, k3)) - _b(clock.c(s) * k3) * J


@nb.core
@torch.no_grad()
def eulerian_target(f, Dbar, x, t, k3, k1=None, clock=None, jvp=jvp_forward, frozen=None):
    """Algorithm 4: target = (1 - kappa) Dbar(x) + kappa f(x, sigma) + kappa J,
    where J is the derivative of f at (x, sigma) in the direction (Dbar(x) - x, -c)."""
    s = clock.kappa_inv(k3)
    Dx = Dbar(x, t)
    u = Dx - x
    if frozen is not None:
        u = u.masked_fill(frozen.unsqueeze(-1), 0.0)       # clean positions do not move
    d, J = jvp(_in_sigma(f), x, s, u, -clock.c(s))
    return (1 - _b(k3)) * Dx + _b(k3) * (d + J)


@nb.core
@torch.no_grad()
def lagrangian_target_positions(f, Dbar, x, k, jvp=jvp_forward):
    """Lagrangian target for one jump fraction per position, k [B, L] (exponential clock: sigma = kappa).
    Along the path a -> a k, a in [0, 1], every position moves linearly in its own time, and
        target = Dbar(x') - (1 - k^l) d/da delta_hat(x, a k) at a = 1,     x'^l = (1 - k^l) x^l + k^l delta_hat^l(x, k).
    A position that goes to the end in this jump (k^l = 1) is left where it is in x': its target is the teacher's
    prediction for it given where the other positions land, and the teacher never reads the model's own output as a
    finished token. With the same k < 1 at every position this is Algorithm 3."""
    d, J = jvp(_in_sigma(f), x, k, torch.zeros_like(x), k)
    km = _b(torch.where(k >= 1 - 1e-6, torch.zeros_like(k), k))
    return Dbar((1 - km) * x + km * d, k[:, 0]) - _b(1 - k) * J


TARGETS = {"semigroup": semigroup_target, "lagrangian": lagrangian_target, "eulerian": eulerian_target}


# ----------------------------------------------------------------------------- refinement (Discrete Beckmann Transport Models)

def ot_coupling(x0, tokens):
    """Minibatch optimal transport between noise and data: reorders x0 [B, L, V] to minimise sum_i |x0_i - x1_i|^2.
    Up to terms that do not depend on the pairing, the cost of (i, j) is -2 sum_l x0_i[l, tokens_j[l]]."""
    from scipy.optimize import linear_sum_assignment
    pos = torch.arange(tokens.shape[1], device=tokens.device)
    gain = torch.stack([x0[:, pos, tok].sum(-1) for tok in tokens], dim=1)       # [B (noise), B (data)]
    rows, cols = linear_sum_assignment(-gain.double().cpu().numpy())
    perm = torch.empty(len(rows), dtype=torch.long)
    perm[torch.as_tensor(cols)] = torch.as_tensor(rows)
    return x0[perm.to(x0.device)]


def commit_step(score, committed, round_index, depth, threshold):
    """Positions committed in one refinement round: every uncommitted position whose score reaches `threshold`, and
    at least ceil(uncommitted / (depth - round + 1)) of the best ones, so that the last round commits everything."""
    unc = ~committed
    n_r = torch.ceil(unc.sum(-1).float() / (depth - round_index + 1)).long().clamp_min(1)        # [B]
    masked = score.masked_fill(committed, -1.0)
    ranked = masked + 1e-6 * torch.rand_like(masked)          # equal scores (an untrained model) are ranked at random
    kth = ranked.sort(dim=-1, descending=True).values.gather(-1, (n_r - 1).unsqueeze(-1))        # n_r-th best score
    return committed | (unc & ((masked >= threshold) | (ranked >= kth)))


@nb.core
@torch.no_grad()
def refinement_rollout(whole_jump, tokens, x0, vocab_size, depth, threshold, commit="data"):
    """Rollout of DBTM (their Algorithm 2, commit mode, used on-policy in training, no gradient).

    whole_jump(x) -> logits of the map to the end (kappa = 1). From the noise x0, each round applies the map, commits
    the positions where it gives the data token a probability >= threshold (and at least an equal share of the rest)
    to the data token, and puts the others back to their own noise. Returns the input state of every round and, for
    every round, the mask of the positions still uncommitted after it (where its loss is taken).

    commit="model" does not use the data: positions are scored by the largest probability and committed to the
    model's own argmax token, as at sampling time. The rollout is then a function of the noise alone."""
    committed = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
    kept = tokens.clone()
    x, states, masks = x0, [], []
    for r in range(1, depth + 1):
        states.append(x)
        p = whole_jump(x).float().softmax(-1)
        if commit == "model":
            q, arg = p.max(-1)
        else:
            q, arg = p.gather(-1, tokens.unsqueeze(-1)).squeeze(-1), tokens      # confidence on the data token
        new = commit_step(q, committed, r, depth, threshold)
        kept = torch.where(new & ~committed, arg, kept)
        committed = new
        x = torch.where(committed.unsqueeze(-1), F.one_hot(kept, vocab_size).float(), x0)
        masks.append(~committed)
    refinement_rollout.final_tokens = kept      # commit="model": the sequence the rollout ends with, a function of x0
    return states, masks


@nb.core
def refinement_round_loss(logits, tokens, mask, target=None, adaptive=None):
    """Loss of one refinement round on its uncommitted positions: cross-entropy to the data (ril-c), or to a soft
    `target` (self-distillation, semi-r). `adaptive` = (c, r) weights each sequence by (loss + c)^(-r), no gradient
    through the weight, as DBTM does for its losses."""
    logp = F.log_softmax(logits.float(), dim=-1)
    per = -logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1) if target is None else -(target * logp).sum(-1)
    per = (per * mask).sum(-1) / mask.sum(-1).clamp_min(1)                  # one value per sequence
    if adaptive is not None:
        per = per * (per.detach() + adaptive[0]) ** (-adaptive[1])
    return per.mean()


def refinement_rollout_loss(whole_jump, tokens, vocab_size, depth, threshold, ot=True, teacher=None, temperature=1.0,
                            adaptive=None):
    """Both refinement-in-loop losses of DBTM in one graph (small batches and tests). Returns (ril_c, semi_r)."""
    x0 = torch.randn(*tokens.shape, vocab_size, device=tokens.device)
    if ot:
        x0 = ot_coupling(x0, tokens)
    states, masks = refinement_rollout(whole_jump, tokens, x0, vocab_size, depth, threshold)
    logits = [whole_jump(x) for x in states]
    ril_c = sum(refinement_round_loss(o, tokens, m, None, adaptive) for o, m in zip(logits, masks)) / depth
    semi_r = ril_c.new_zeros(())
    if teacher is not None:
        with torch.no_grad():
            target = (teacher(states[-1]).float() / temperature).softmax(-1)
        semi_r = sum(refinement_round_loss(o, tokens, m, target, adaptive) for o, m in zip(logits, masks)) / depth
    return ril_c, semi_r
