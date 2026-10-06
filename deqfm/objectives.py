"""The two training objectives, as functions  loss_fn(model, raw, tokens) -> (loss, logs).

`model` is the (possibly DataParallel-wrapped) network used for every forward pass that receives a gradient.
`raw` is the same network unwrapped; it is used to build targets, without gradient.
"""
import torch

from .losses import (TARGETS, diagonal_data_loss, jvp_forward, make_jvp_fd, sample_diag_time, sample_state,
                     soft_cross_entropy)


def make_field_loss(cfg, vocab_size):
    """Autonomous field only: the denoiser D_hat = delta_hat_0, trained with the weighted cross-entropy."""
    L = cfg.loss

    def loss_fn(model, raw, tokens):
        t, w = sample_diag_time(tokens.shape[0], L.diag_time, L.t_eps, tokens.device)
        x = sample_state(tokens, vocab_size, t)
        loss = diagonal_data_loss(model(x, torch.zeros_like(t)), tokens, w)
        return loss, {"loss": loss.item()}

    return loss_fn


def make_flow_map_loss(cfg, vocab_size, clock, teacher=None):
    """Algorithm 1: diagonal loss + alpha * off-diagonal loss.

    teacher=None : training from data. The diagonal uses the data and Dbar is the model's own delta_hat_0.
    teacher given: distillation. The diagonal matches the frozen teacher and Dbar is the teacher.
    """
    L = cfg.loss
    target_fn = TARGETS[L.off_diagonal]
    forward_mode = L.jvp == "forward"
    jvp = jvp_forward if forward_mode else make_jvp_fd(L.fd_eps)

    def probs(net, **kw):
        return lambda y, s: net(y, s, **kw).float().softmax(dim=-1)

    def loss_fn(model, raw, tokens):
        if L.split_batch and tokens.shape[0] >= 2:
            half = tokens.shape[0] // 2
            tok_diag, tok_off = tokens[:half], tokens[half:]
        else:
            tok_diag = tok_off = tokens
        zeros = lambda y: torch.zeros(y.shape[0], device=y.device)

        # diagonal (sigma = 0)
        if teacher is None:
            t, w = sample_diag_time(tok_diag.shape[0], L.diag_time, L.t_eps, tokens.device)
            x = sample_state(tok_diag, vocab_size, t)
            loss_diag = diagonal_data_loss(model(x, zeros(x)), tok_diag, w)
        else:
            x = sample_state(tok_diag, vocab_size, torch.rand(tok_diag.shape[0], device=tokens.device))
            with torch.no_grad():
                target = probs(teacher)(x, zeros(x))
            loss_diag = soft_cross_entropy(model(x, zeros(x)), target)

        # off-diagonal (sigma > 0)
        x = sample_state(tok_off, vocab_size, torch.rand(tok_off.shape[0], device=tokens.device))
        f = probs(raw, manual_attn=forward_mode)
        Dbar_net = probs(teacher if teacher is not None else raw)
        Dbar = lambda y: Dbar_net(y, zeros(y))
        sigma, target = target_fn(f, Dbar, x, clock, jvp)
        loss_off = soft_cross_entropy(model(x, sigma), target)

        loss = loss_diag + L.alpha * loss_off
        return loss, {"loss": loss.item(), "diag": loss_diag.item(), "off": loss_off.item()}

    return loss_fn
