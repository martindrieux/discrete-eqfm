"""Clocks.

A clock r maps autonomous time tau in [0, inf) to a duration label sigma in [0, 1),
with r(0) = 0, r'(0) = 1, r increasing and r(tau) -> 1.

The algorithms only need two functions of sigma:

    c(sigma)     = r'(tau) at sigma = r(tau)        clock speed
    kappa(sigma) = 1 - exp(-r^{-1}(sigma))          weight of the denoised point in one jump

and the inverse of kappa. A jump of duration sigma is

    Phi_sigma(x) = (1 - kappa(sigma)) x + kappa(sigma) delta_sigma(x).

All methods take and return tensors; sigma = 1 (the terminal map) is allowed.
"""
import torch


class ExponentialClock:
    """r(tau) = 1 - exp(-tau).  kappa(sigma) = sigma, c(sigma) = 1 - sigma."""

    name = "exponential"

    def c(self, sigma):
        return 1 - sigma

    def kappa(self, sigma):
        return sigma

    def kappa_inv(self, kappa):
        return kappa


class RationalClock:
    """r(tau) = tau / (1 + tau).  kappa(sigma) = 1 - exp(-sigma / (1 - sigma)), c(sigma) = (1 - sigma)^2."""

    name = "rational"

    def c(self, sigma):
        return (1 - sigma) ** 2

    def kappa(self, sigma):
        return 1 - torch.exp(-sigma / (1 - sigma))  # sigma = 1 gives exp(-inf) = 0

    def kappa_inv(self, kappa):
        tau = -torch.log1p(-kappa)  # kappa = 1 gives tau = inf
        return 1 - 1 / (1 + tau)


CLOCKS = {"exponential": ExponentialClock, "rational": RationalClock}


def build_clock(name):
    if name not in CLOCKS:
        raise ValueError(f"unknown clock '{name}', choose from {sorted(CLOCKS)}")
    return CLOCKS[name]()
