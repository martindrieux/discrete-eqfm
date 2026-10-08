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
def partial_context(t, seq_len):
    """Partial-context interpolant of Discrete Beckmann Transport Models: per sample, a fraction f ~ U[0, 1] of the
    positions (each with probability f) is clean context, t = 1; the others keep the sample's time. Returns [B, L]."""
    f = torch.rand(t.shape[0], 1, device=t.device)
    clean = torch.rand(t.shape[0], seq_len, device=t.device) < f
    return torch.where(clean, torch.ones_like(f), t.view(-1, 1).expand(-1, seq_len))


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
def soft_cross_entropy(logits, target):
    """- target . log softmax(logits), averaged over positions and batch.

    When the rows of `target` sum to one, the gradient with respect to the logits is softmax(logits) - target.
    """
    return -(target.float() * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


@nb.core
def diagonal_data_loss(logits, tokens, weight):
    """Weighted cross-entropy against the data: mean over the batch of weight * (- x_1 . log delta_hat_0(I_t)) / L."""
    ce = F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten(), reduction="none").view(tokens.shape)
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


# ----------------------------------------------------------------------------- jump sampling

@nb.core
def sample_jumps(batch, mode, clock, device, need_mid, reparam=None, boundary_prob=0.0, stratified=False, end_prob=0.0):
    """State time t, jump fraction k3 and (for the semigroup loss) first-leg fraction k1, each of shape [batch].

    notes   : t ~ U[0, 1]. Semigroup: sigma_3 with density 2 sigma_3, sigma_1 ~ U[0, sigma_3]. Otherwise sigma ~ U[0, 1].
    endpoint: t ~ U[0, 1] and kappa = 1 always: the map to the end of the flow only (not for the semigroup loss).
    reparam : as FMLM, in the reparameterised time tau: h ~ U[0, 1], tau_s ~ U[0, 1 - h], tau_t = tau_s + h, midpoint
              (tau_s + tau_t) / 2; with probability `boundary_prob` the pair is (0, 1), a whole one-step jump;
              with probability `end_prob` it is (tau_s, 1) with tau_s ~ U[0, 1], a jump to the end from any time.
    """
    if mode == "notes":
        t = uniform01(batch, device, stratified)
        if need_mid:
            s3 = torch.rand(batch, device=device).sqrt()
            s1 = torch.rand(batch, device=device) * s3
            return t, clock.kappa(s3), clock.kappa(s1)
        return t, clock.kappa(torch.rand(batch, device=device)), None
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
def semigroup_target(f, Dbar, x, t, k3, k1, clock=None, jvp=None):
    """Algorithm 2: target = gamma f(x, k1) + (1 - gamma) f(x', k2), with x' the state after the first leg."""
    k2 = ((k3 - k1) / (1 - k1).clamp_min(1e-8)).clamp(0, 1)
    gamma = k1 * (1 - k2) / k3.clamp_min(1e-8)
    d1 = f(x, t, k1)
    x_mid = (1 - _b(k1)) * x + _b(k1) * d1
    d2 = f(x_mid, land(t, k1), k2)
    return _b(gamma) * d1 + (1 - _b(gamma)) * d2


def _in_sigma(f):
    """f as a function of (x, sigma), for derivatives with respect to the duration label."""
    if not hasattr(f, "in_sigma"):
        raise ValueError("the Lagrangian and Eulerian losses need model.cond = duration or state_time")
    return f.in_sigma


@nb.core
@torch.no_grad()
def lagrangian_target(f, Dbar, x, t, k3, k1=None, clock=None, jvp=jvp_forward):
    """Algorithm 3: target = Dbar(x') - c kappa d/dsigma f(x, sigma), with x' the state after the jump."""
    s = clock.kappa_inv(k3)
    d, J = jvp(_in_sigma(f), x, s, torch.zeros_like(x), torch.ones_like(s))
    x_end = (1 - _b(k3)) * x + _b(k3) * d
    return Dbar(x_end, land(t, k3)) - _b(clock.c(s) * k3) * J


@nb.core
@torch.no_grad()
def eulerian_target(f, Dbar, x, t, k3, k1=None, clock=None, jvp=jvp_forward):
    """Algorithm 4: target = (1 - kappa) Dbar(x) + kappa f(x, sigma) + kappa J,
    where J is the derivative of f at (x, sigma) in the direction (Dbar(x) - x, -c)."""
    s = clock.kappa_inv(k3)
    Dx = Dbar(x, t)
    d, J = jvp(_in_sigma(f), x, s, Dx - x, -clock.c(s))
    return (1 - _b(k3)) * Dx + _b(k3) * (d + J)


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
def refinement_rollout_loss(whole_jump, tokens, vocab_size, depth, threshold, ot=True, teacher=None, temperature=1.0):
    """Refinement-in-loop losses of DBTM (their Algorithm 3, commit mode).

    whole_jump(x) -> logits of the map to the end (kappa = 1). The model is rolled out for `depth` rounds from noise:
    each round the map is applied, the positions where it gives the data token a probability >= threshold (and at
    least an equal share of the rest) are committed to the data token, and the others go back to their own noise.
    Every round is trained to recover the data on its uncommitted positions (ril-c). With a `teacher` (a frozen copy
    of the model), every round is also trained towards the teacher's sharpened output at the last round (semi-r).
    Returns (ril_c, semi_r)."""
    x1 = F.one_hot(tokens, vocab_size).float()
    x0 = torch.randn_like(x1)
    if ot:
        x0 = ot_coupling(x0, tokens)
    committed = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
    x, logits, masks = x0, [], []
    for r in range(1, depth + 1):
        last_state = x
        out = whole_jump(x)
        logits.append(out)
        with torch.no_grad():
            q = out.float().softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)     # confidence on the data token
            committed = commit_step(q, committed, r, depth, threshold)
            x = torch.where(committed.unsqueeze(-1), x1, x0)
        masks.append(~committed)       # as in DBTM: the loss of round r is on the positions still uncommitted after it

    def masked_mean(per_position, mask):
        return (per_position * mask).sum() / mask.sum().clamp_min(1)

    ril_c = sum(masked_mean(F.cross_entropy(o.float().flatten(0, 1), tokens.flatten(), reduction="none").view(tokens.shape), m)
                for o, m in zip(logits, masks)) / depth
    semi_r = ril_c.new_zeros(())
    if teacher is not None:
        with torch.no_grad():
            target = (teacher(last_state).float() / temperature).softmax(-1)
        semi_r = sum(masked_mean(-(target * F.log_softmax(o.float(), -1)).sum(-1), m) for o, m in zip(logits, masks)) / depth
    return ril_c, semi_r
