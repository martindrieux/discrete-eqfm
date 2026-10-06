"""Time reparameterisation of FLM (decoding-error clock) and the noise level read off a state.

tau(t) = 1 - V / (V - 1) * P_e(t), where P_e(t) is the probability that the argmax of one position of
I_t = (1 - t) x_0 + t x_1 is not the data token. Uniform steps in tau are uniform steps in decoding progress.
For V = 30522 almost all of the progress happens for t in [0.65, 0.9].

The table is built as in flm/utils.py (Gauss-Hermite quadrature, 10000 knots, cubic splines), so tau <-> t agree
with the released checkpoints to spline accuracy.
"""
import functools

import numpy as np
import torch


from .track import nb

def tau_of_t_exact(t, vocab_size, n_gh=100):
    from numpy.polynomial.hermite import hermgauss
    from scipy.special import log_ndtr
    t = np.asarray(t, dtype=np.float64)
    m = t / np.maximum(1.0 - t, 1e-12)
    x, w = hermgauss(n_gh)
    z = np.sqrt(2.0) * x
    q = np.sum(w / np.sqrt(np.pi) * np.exp((vocab_size - 1) * log_ndtr(z[None, :] + m[:, None])), axis=-1)
    tau = vocab_size / (vocab_size - 1.0) * (q - 1.0 / vocab_size) + (t - 1) * 1e-10
    return np.clip(tau, 0.0, 1.0)


class Reparam:
    def __init__(self, vocab_size, n_points=10000):
        from scipy.interpolate import CubicSpline
        t = np.linspace(0.0, 1.0, n_points)
        tau = tau_of_t_exact(t, vocab_size)
        self._t2tau = CubicSpline(t, tau)
        tau_u, idx = np.unique(tau, return_index=True)
        self._tau2t = CubicSpline(tau_u, t[idx])
        self.vocab_size = vocab_size
        self._table = torch.from_numpy(tau).float()      # tau at uniform knots in t, for the differentiable version
        self._tables = {}

    @staticmethod
    def _apply(spline, v):
        out = np.clip(spline(v.detach().double().cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(out).to(v.device, dtype=v.dtype)

    def tau(self, t):
        """t -> tau, tensors of any shape (no gradient)."""
        return self._apply(self._t2tau, t)

    def t(self, tau):
        """tau -> t."""
        return self._apply(self._tau2t, tau)

    def tau_diff(self, t):
        """t -> tau by linear interpolation of the table: differentiable in t (forward and reverse mode)."""
        key = (t.device, t.dtype)
        if key not in self._tables:
            self._tables[key] = self._table.to(device=t.device, dtype=t.dtype)
        table = self._tables[key]
        pos = t.clamp(0, 1) * (table.numel() - 1)
        i = pos.detach().floor().long().clamp(max=table.numel() - 2)
        return table[i] + (pos - i) * (table[i + 1] - table[i])

    def grid(self, steps):
        """Sampling grid t_0 = 0 < ... < t_K = 1, uniform in tau, as a list of floats."""
        return [float(v) for v in self._apply(self._tau2t, torch.linspace(0, 1, steps + 1, dtype=torch.float64))]


@functools.lru_cache(maxsize=None)
def get_reparam(vocab_size):
    return Reparam(vocab_size)


@nb.core
def state_time(x):
    """Noise level of a state, from its norm alone. x: [B, L, V] -> t_hat: [B].

    For x = (1 - t) x_0 + t x_1 with x_0 ~ N(0, I) and one-hot x_1, the mean of |x^l|^2 over positions is
    (1 - t)^2 V + t^2 up to fluctuations of relative size 1 / sqrt(L V). Solving for t gives the root below.
    A flow-map state (1 - kappa) x + kappa delta keeps the same noise component, so the estimate follows the jump.
    """
    V = x.shape[-1]
    m = torch.linalg.vector_norm(x.float(), dim=-1).pow(2).mean(-1)   # [B]; no vocabulary-sized temporary
    disc = (V * V - (V + 1) * (V - m)).clamp_min(0.0)
    return ((V - disc.sqrt()) / (V + 1)).clamp(0.0, 1.0)
