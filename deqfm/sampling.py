"""Sampling (Algorithm 5 of the notes).

Each step is  x <- (1 - kappa_k) x + kappa_k p(x),  with the last kappa equal to 1.

flow_map=True : p(x) is the learned jump of fraction kappa_k: the learned flow map.
flow_map=False: p(x) is the denoiser: integrates dx/dtau = D(x) - x holding D fixed over each step (the Euler
                sampler of FLM when the grid is in the same time variable).

The steps are given as a time grid 0 = t_0 < ... < t_K = 1, with kappa_k = (t_{k+1} - t_k) / (1 - t_k):
    uniform : t_k = k / K, i.e. kappa_k = 1 / (K - k): every step removes the same amount of noise level
    reparam : uniform in the decoding-error clock of FLM (deqfm/reparam.py)
"""
import torch

from .nets import make_fn, probs_fn
from .reparam import get_reparam


from .track import nb

def drafted(model, cond, clock):
    """f(x, t, kappa, draft) -> probabilities. A model with self-conditioning (ELF) is told the draft, the output of the
    previous call (None at the first one); other models ignore it."""
    sc = getattr(model, "sc_embed", None) is not None

    def f(x, t, kappa, draft=None):
        kw = {"sc": draft} if sc and draft is not None else {}
        return probs_fn(make_fn(model, cond, clock, model.vocab_size, **kw))(x, t, kappa)
    return f


def uniform_kappas(steps):
    """kappa_k = 1 / (K - k): every step removes the same amount of noise level. The last one is 1."""
    return [1.0 / (steps - k) for k in range(steps)]


@nb.core
def time_grid(steps, kind="uniform", vocab_size=None):
    if kind == "uniform":
        return [k / steps for k in range(steps + 1)]
    if kind == "reparam":
        return get_reparam(vocab_size).grid(steps)
    raise ValueError(f"unknown grid '{kind}'")


@nb.core
@torch.no_grad()
def sample_grid(model, cond, clock, num, grid, flow_map=True, churn=0.0, device="cpu", return_state=False, noise_scale=1.0, temp=0.0):
    """Token ids [num, seq_len] from a time grid. `churn` is the gamma of the FMLM sampler: each step jumps a
    little past its target time and fresh noise brings the state back to it (0: deterministic).
    temp > 0: the tokens are drawn from p^(1 / temp) of the last step instead of its argmax."""
    f = drafted(model, cond, clock)
    x = noise_scale * torch.randn(num, model.seq_len, model.vocab_size, device=device)   # scale != 1: entropy knob
    p = None
    for t0, t1 in zip(grid[:-1], grid[1:]):
        t = torch.full((num,), float(t0), device=device)
        last = t1 >= 1.0
        noise_now = (1 - t1) * (1 - churn ** 2) ** 0.5       # noise level right after the jump
        kappa = 1.0 if last else 1 - noise_now / (1 - t0)
        k = torch.full((num,), float(kappa), device=device)
        p = f(x, t, k if flow_map else torch.zeros_like(k), p)
        if last:
            x = p
            break
        x = (1 - kappa) * x + kappa * p
        if churn > 0:
            x = x + (noise_now - (1 - t1)) * p + churn * (1 - t1) * torch.randn_like(x)
    return x if return_state else draw_tokens(x, temp)[1]


def draw_tokens(p, temp=0.0):
    """(confidence, token) per position: the argmax of p, or for temp > 0 a draw from p^(1 / temp)."""
    if temp <= 0:
        return p.max(dim=-1)
    q = p.clamp_min(1e-30).log().div(temp).softmax(-1)
    tok = torch.multinomial(q.flatten(0, -2), 1).view(p.shape[:-1])
    return p.gather(-1, tok.unsqueeze(-1)).squeeze(-1), tok


@nb.core
@torch.no_grad()
def sample_remask(model, cond, clock, num, rounds, threshold=0.9, device="cpu", noise_scale=1.0, keep=0.0, recommit=False,
                  temp=0.0, flow_map=True):
    """Commit-and-renoise sampler with three options taken from masked diffusion samplers. One evaluation per round.

    Round r of R: p = whole jump from the state (kappa = 1; the denoiser if flow_map is False), g^l = max_v p^l_v.
        commit   : positions with g^l >= threshold, and at least enough of the best ones to hold ceil(L r / R) in all,
                   are set to their token (argmax, or a draw from p^(1 / temp) if temp > 0).
        recommit : the L r / R positions are chosen again each round among all positions, a committed position being
                   ranked by the confidence it had when committed: a low-confidence commit can be undone.
        keep     : an uncommitted position is put at noise level t_r = keep r / R of its current guess,
                   x^l = (1 - t_r) c eps + t_r p^l, eps fresh (keep = 0: pure fresh noise, the sampler of DBTM).
    The network is told time t_r (a network that is told no time ignores it); the last round commits everything."""
    f = drafted(model, cond, clock)
    V, Lq = model.vocab_size, model.seq_len
    kap = torch.ones(num, device=device) if flow_map else torch.zeros(num, device=device)
    x = noise_scale * torch.randn(num, Lq, V, device=device)
    committed = torch.zeros(num, Lq, dtype=torch.bool, device=device)
    tokens = torch.zeros(num, Lq, dtype=torch.long, device=device)
    held = torch.zeros(num, Lq, device=device)               # confidence of each committed position when committed
    t_r, p = 0.0, None
    for r in range(1, rounds + 1):
        p = f(x, torch.full((num,), float(t_r), device=device), kap, p)
        conf, tok = draw_tokens(p, temp)
        score = torch.where(committed, held, conf)
        n_r = -(-Lq * r // rounds)                           # ceil(L r / R) positions held after this round
        if recommit:
            ranked = score + 1e-6 * torch.rand_like(score)
            kth = ranked.sort(dim=-1, descending=True).values[:, n_r - 1:n_r]
            new = (score >= threshold) | (ranked >= kth)
        else:
            ranked = conf.masked_fill(committed, -1.0) + 1e-6 * torch.rand_like(conf)
            need = (n_r - committed.sum(-1, keepdim=True)).clamp_min(1)
            kth = ranked.sort(dim=-1, descending=True).values.gather(-1, (need - 1).clamp(max=Lq - 1))
            new = committed | (conf >= threshold) | (ranked >= kth)
        if r == rounds:
            new = torch.ones_like(committed)
        fresh = new & ~committed
        tokens, held = torch.where(fresh, tok, tokens), torch.where(fresh, conf, held)
        committed = new
        if r == rounds:
            break
        t_r = keep * r / rounds
        soft = (1 - t_r) * noise_scale * torch.randn_like(x) + t_r * p
        x = torch.where(committed.unsqueeze(-1), torch.nn.functional.one_hot(tokens, V).to(x.dtype), soft)
    return tokens


@nb.core
@torch.no_grad()
def sample_refine(model, cond, clock, num, rounds, threshold=0.9, device="cpu", noise_scale=1.0, fresh=True, rep_lambda=0.0,
                  flow_steps=1, churn=0.0):
    """Sampler with refinement of DBTM (their Algorithm 2, commit mode, softmax confidence, fresh noise).

    Each round applies the whole jump (kappa = 1) to the current state, commits the positions whose largest
    probability reaches `threshold` (and at least an equal share of the rest) to their argmax token, and puts fresh
    noise on the others. The last round commits everything. One network evaluation per round; one round is the
    ordinary one-step sample. rep_lambda > 0 is the repetition exponent of DBTM: on uncommitted positions the
    probability of token i is divided by (1 + n_i)^lambda, n_i the number of committed positions holding i.
    flow_steps N > 1 (posterior refinement of FMLM+): in each round the uncommitted positions take N jumps on the
    decoding-clock grid from their noise instead of one whole jump, the committed ones held; N evaluations per round.
    churn: the gamma of the FMLM sampler inside these jumps."""
    from .losses import commit_step
    inner = time_grid(flow_steps, "reparam", model.vocab_size) if flow_steps > 1 else None
    f = drafted(model, cond, clock)
    p = None
    ones, zeros = torch.ones(num, device=device), torch.zeros(num, device=device)
    x = x0 = noise_scale * torch.randn(num, model.seq_len, model.vocab_size, device=device)
    committed = torch.zeros(num, model.seq_len, dtype=torch.bool, device=device)
    tokens = torch.zeros(num, model.seq_len, dtype=torch.long, device=device)
    for r in range(1, rounds + 1):
        if inner is None:
            p = f(x, zeros, ones, p)
        else:
            for t0, t1 in zip(inner[:-1], inner[1:]):
                noise_now = (1 - t1) * (1 - churn ** 2) ** 0.5
                kappa = 1.0 if t1 >= 1.0 else 1 - noise_now / (1 - t0)
                p = f(x, zeros + float(t0), zeros + float(kappa), p)
                if t1 >= 1.0:
                    break
                moved = (1 - kappa) * x + kappa * p
                if churn > 0:
                    moved = moved + (noise_now - (1 - t1)) * p + churn * (1 - t1) * noise_scale * torch.randn_like(x)
                x = torch.where(committed.unsqueeze(-1), x, moved)
        if rep_lambda > 0 and committed.any():
            counts = torch.zeros(num, model.vocab_size, device=device)
            counts.scatter_add_(1, tokens, committed.float())
            p = p / (1 + counts).pow(rep_lambda).unsqueeze(1)
        conf, arg = p.max(dim=-1)
        new = commit_step(conf, committed, r, rounds, threshold) & ~committed
        tokens = torch.where(new, arg, tokens)
        committed = committed | new
        if committed.all():
            break
        noise = noise_scale * torch.randn_like(x) if fresh else x0      # fresh=False: uncommitted positions keep their noise
        x = torch.where(committed.unsqueeze(-1), torch.nn.functional.one_hot(tokens, model.vocab_size).float(), noise)
    return tokens


@nb.core
@torch.no_grad()
def sample_adaptive(model, cond, clock, num, grid, h=0.9, score="maxprob", when="post", flow_map=True, device="cpu",
                    renoise=False, noise_scale=1.0):
    """Sampler with committed positions (Section 7.2 of the notes, binary version). Returns (tokens, evaluations).

    The state follows the time grid as in `sample_grid`, but a position whose confidence exceeds `h` is committed:
    it is set to the one-hot vector of its most likely token and no longer moves. The other positions keep
    their noise and take the jump.
        when = post: the confidence is read from the output of the jump itself (one evaluation per step)
        when = pre : before each jump after the first, the map to the end is evaluated at the current state, positions
                     are committed from it, and the jump is taken from the updated state (two evaluations per step)
    flow_map=False: the denoiser is used for both (Euler steps). renoise: uncommitted positions are put back on the
    interpolant between their own noise and the model's current guess (not used by default)."""
    from .losses import confidence
    f = probs_fn(make_fn(model, cond, clock, model.vocab_size))
    V = model.vocab_size
    x = noise_scale * torch.randn(num, model.seq_len, V, device=device)   # scale != 1: entropy knob
    committed = torch.zeros(num, model.seq_len, dtype=torch.bool, device=device)
    evals = 0

    def commit(p, x, committed):
        new = (confidence(p, score) > h) & ~committed
        x = torch.where(new.unsqueeze(-1), torch.nn.functional.one_hot(p.argmax(-1), V).to(x.dtype), x)
        return x, committed | new

    for i, (t0, t1) in enumerate(zip(grid[:-1], grid[1:])):
        t = torch.full((num,), float(t0), device=device)
        last = t1 >= 1.0
        kappa = 1.0 if last else (t1 - t0) / (1 - t0)
        k = torch.full((num,), float(kappa), device=device)
        if when == "pre" and i > 0:
            x, committed = commit(f(x, t, torch.ones_like(k) if flow_map else torch.zeros_like(k)), x, committed)
            evals += 1
        p = f(x, t, k if flow_map else torch.zeros_like(k))
        evals += 1
        if last:
            x = torch.where(committed.unsqueeze(-1), x, p)
            break
        moved = (1 - kappa) * x + kappa * p
        x = torch.where(committed.unsqueeze(-1), x, moved)
        if when == "post":
            x, committed = commit(p, x, committed)
    return x.argmax(dim=-1), evals


@nb.core
@torch.no_grad()
def sample_adaptive_speed(model, cond, clock, num, grid, beta=4.0, score="maxprob", device="cpu"):
    """Sampler with one time per position (Section 7.2 of the notes, continuous version), for a denoiser.

    Each step of the grid has length dt. With p = D(x) and g^l its confidence, position l advances from t^l to
    t'^l = min(1, t^l + dt (1 + beta g^l)) by x^l <- (1 - k) x^l + k p^l, k = (t'^l - t^l) / (1 - t^l). A position
    that reaches time 1 is set to the one-hot vector of its most likely token and no longer moves. The last step
    takes every position to time 1. One evaluation per step."""
    from .losses import confidence
    f = probs_fn(make_fn(model, cond, clock, model.vocab_size))
    V = model.vocab_size
    x = torch.randn(num, model.seq_len, V, device=device)
    tl = torch.zeros(num, model.seq_len, device=device)
    for t0, t1 in zip(grid[:-1], grid[1:]):
        t = torch.full((num,), float(t0), device=device)
        p = f(x, t, torch.zeros_like(t))
        new = (tl + (t1 - t0) * (1 + beta * confidence(p, score))).clamp(max=1.0)
        if t1 >= 1.0:
            new = torch.ones_like(tl)
        k = ((new - tl) / (1 - tl).clamp_min(1e-8)).clamp(0, 1).unsqueeze(-1)
        moved = (1 - k) * x + k * p
        done = (new >= 1) & (tl < 1)
        moved = torch.where(done.unsqueeze(-1), torch.nn.functional.one_hot(p.argmax(-1), V).to(x.dtype), moved)
        x = torch.where((tl >= 1).unsqueeze(-1), x, moved)
        tl = new
    return x.argmax(dim=-1)


@nb.core
@torch.no_grad()
def sample_speed_clock(model, cond, clock, num, steps, beta=16.0, power=4.0, score="maxprob", device="cpu", snap=True,
                       floor=0.0, base=1.0, threshold=None, x0=None):
    """Continuous adaptive speed in the decoding-error clock, for a denoiser (Section 7.2 of the notes, equations 31-32).

    Every position has its own progress u^l in [0, 1] (u = tau(t), the decoding-error clock) and noise time t^l = tau^{-1}(u^l).
    One evaluation per step: p = D(x), g^l = confidence of p^l, and
        u^l <- min(1, u^l + (1 / K) psi(g^l)),   psi(g) = base + beta max(g - floor, 0)^power,   base = 1 by default,
        x^l <- (1 - k) x^l + k p^l,              k = (t^l_new - t^l) / (1 - t^l).
    psi(0) = 1: a position the model is unsure about moves on the ordinary K-step grid; a confident one moves faster.
    A position that reaches u^l = 1 is set to the one-hot vector of its most likely token (`snap`) and stops.
    beta = 0 is the plain Euler sampler on the decoding-clock grid. base < 1: unsure positions move slower than the
    grid and wait for the others; whatever is left at the last step is taken to the end in one jump.
    `threshold` h replaces the law by the binary one, psi(g) = base + beta 1[g > h] (beta large: commit at once)."""
    from .losses import confidence
    f = probs_fn(make_fn(model, cond, clock, model.vocab_size))
    V, rep = model.vocab_size, get_reparam(model.vocab_size)
    x = torch.randn(num, model.seq_len, V, device=device) if x0 is None else x0      # x0: start from a given noise
    u = torch.zeros(num, model.seq_len, device=device)
    t = torch.zeros_like(u)
    zeros = torch.zeros(num, device=device)
    for k in range(steps):
        # a network that is told the time gets the time of the plain K-step grid (the positions it is unsure about)
        p = f(x, zeros + float(rep.t(torch.tensor([k / steps]))), zeros)
        g = (confidence(p, score) - floor).clamp_min(0.0)
        boost = g.pow(power) if threshold is None else (g > threshold).float()     # threshold: the binary law, for comparison
        new_u = (u + (base + beta * boost) / steps).clamp(max=1.0)
        if k == steps - 1:
            new_u = torch.ones_like(u)
        new_t = torch.where(new_u >= 1, torch.ones_like(t), rep.t(new_u))
        kappa = ((new_t - t) / (1 - t).clamp_min(1e-8)).clamp(0, 1).unsqueeze(-1)
        moved = (1 - kappa) * x + kappa * p
        done = (new_u >= 1) & (u < 1)
        if snap:
            moved = torch.where(done.unsqueeze(-1), torch.nn.functional.one_hot(p.argmax(-1), V).to(x.dtype), moved)
        x = torch.where((u >= 1).unsqueeze(-1), x, moved)
        u, t = new_u, new_t
    return x.argmax(dim=-1)


@nb.core
@torch.no_grad()
def sample_speed_map(model, cond, clock, num, steps, beta=4.0, power=1.0, device="cpu", noise_scale=1.0, shared=False):
    """Adaptive-speed sampler for a flow map with one duration per position (trained with loss.position_speed).

    Every position has its own progress u^l (decoding-error clock) and time t^l = tau^{-1}(u^l). One evaluation per step:
        u'^l = min(1, u^l + (1 / K)(1 + beta (g^l)^power)),     kappa^l = (tau^{-1}(u'^l) - t^l) / (1 - t^l),
        p = delta_hat(x, kappa),     x^l <- (1 - kappa^l) x^l + kappa^l p^l,     g^l <- max_v p^l_v,
    g = 0 before the first step. A position that reaches u^l = 1 becomes the one-hot vector of its most likely token
    and stops; the last step takes every position to the end. beta = 0: the plain K-step sampler on the decoding-clock grid.
    shared: the network is told one duration per sequence (the mean kappa of the positions still moving), while each
    position still moves by its own kappa^l: separates the effect of the per-position input from the update rule."""
    f = probs_fn(make_fn(model, cond, clock, model.vocab_size))
    V, rep = model.vocab_size, get_reparam(model.vocab_size)
    x = noise_scale * torch.randn(num, model.seq_len, V, device=device)
    u = torch.zeros(num, model.seq_len, device=device)
    t, g = torch.zeros_like(u), torch.zeros_like(u)
    zeros = torch.zeros(num, device=device)
    for k in range(steps):
        new_u = torch.ones_like(u) if k == steps - 1 else (u + (1 + beta * g.pow(power)) / steps).clamp(max=1.0)
        new_t = torch.where(new_u >= 1, torch.ones_like(t), rep.t(new_u))
        kappa = torch.where(u >= 1, torch.zeros_like(u), ((new_t - t) / (1 - t).clamp_min(1e-8)).clamp(0, 1))
        if shared:
            moving = (u < 1).float()
            told = (kappa * moving).sum(-1, keepdim=True) / moving.sum(-1, keepdim=True).clamp_min(1)
            p = f(x, zeros, told.expand_as(kappa))
        else:
            p = f(x, zeros, kappa)
        moved = (1 - kappa.unsqueeze(-1)) * x + kappa.unsqueeze(-1) * p
        moved = torch.where((new_u >= 1).unsqueeze(-1), torch.nn.functional.one_hot(p.argmax(-1), V).to(x.dtype), moved)
        x = torch.where((u >= 1).unsqueeze(-1), x, moved)
        u, t, g = new_u, new_t, p.max(-1).values
    return x.argmax(dim=-1)


@torch.no_grad()
def sample(model, clock, num, steps, flow_map=True, kappas=None, device="cpu", cond="duration", grid="uniform"):
    """Returns token ids [num, seq_len]. `kappas` overrides the grid (duration-conditioned models only)."""
    if kappas is None:
        return sample_grid(model, cond, clock, num, time_grid(steps, grid, model.vocab_size), flow_map, device=device)
    x = torch.randn(num, model.seq_len, model.vocab_size, device=device)
    for kappa in kappas:
        k = torch.full((num,), float(kappa), device=device)
        sigma = clock.kappa_inv(k) if flow_map else torch.zeros_like(k)
        p = model(x, sigma).float().softmax(dim=-1)
        x = (1 - kappa) * x + kappa * p
    return x.argmax(dim=-1)


def unigram_entropy(tokens):
    """Mean over samples of the entropy (nats) of the token histogram of each sequence."""
    ent = []
    for row in tokens:
        p = torch.bincount(row).float()
        p = p[p > 0] / p.sum()
        ent.append(-(p * p.log()).sum())
    return torch.stack(ent).mean().item()
