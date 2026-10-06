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


def uniform_kappas(steps):
    """kappa_k = 1 / (K - k): every step removes the same amount of noise level. The last one is 1."""
    return [1.0 / (steps - k) for k in range(steps)]


def time_grid(steps, kind="uniform", vocab_size=None):
    if kind == "uniform":
        return [k / steps for k in range(steps + 1)]
    if kind == "reparam":
        return get_reparam(vocab_size).grid(steps)
    raise ValueError(f"unknown grid '{kind}'")


@torch.no_grad()
def sample_grid(model, cond, clock, num, grid, flow_map=True, churn=0.0, device="cpu", return_state=False):
    """Token ids [num, seq_len] from a time grid. `churn` is the gamma of the FMLM sampler: each step jumps a
    little past its target time and fresh noise brings the state back to it (0: deterministic)."""
    f = probs_fn(make_fn(model, cond, clock, model.vocab_size))
    x = torch.randn(num, model.seq_len, model.vocab_size, device=device)
    for t0, t1 in zip(grid[:-1], grid[1:]):
        t = torch.full((num,), float(t0), device=device)
        last = t1 >= 1.0
        noise_now = (1 - t1) * (1 - churn ** 2) ** 0.5       # noise level right after the jump
        kappa = 1.0 if last else 1 - noise_now / (1 - t0)
        k = torch.full((num,), float(kappa), device=device)
        p = f(x, t, k if flow_map else torch.zeros_like(k))
        if last:
            x = p
            break
        x = (1 - kappa) * x + kappa * p
        if churn > 0:
            x = x + (noise_now - (1 - t1)) * p + churn * (1 - t1) * torch.randn_like(x)
    return x if return_state else x.argmax(dim=-1)


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
