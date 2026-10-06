"""Sampling (Algorithm 5 of the notes).

Each step is  x <- (1 - kappa_k) x + kappa_k p(x),  with the last kappa equal to 1.

flow_map=True : p(x) = delta_hat_{sigma_k}(x) with sigma_k = kappa^{-1}(kappa_k): the learned flow map.
flow_map=False: p(x) = delta_hat_0(x) = D_hat(x): integrates dx/dtau = D(x) - x holding D fixed over each step.
"""
import torch


def uniform_kappas(steps):
    """kappa_k = 1 / (K - k + 1): every step removes the same amount of noise level. The last one is 1."""
    return [1.0 / (steps - k) for k in range(steps)]


@torch.no_grad()
def sample(model, clock, num, steps, flow_map=True, kappas=None, device="cpu"):
    """Returns token ids [num, seq_len]."""
    kappas = uniform_kappas(steps) if kappas is None else kappas
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
