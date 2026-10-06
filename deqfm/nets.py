"""What the network is told about a state and a jump: `model.cond`.

Every loss and sampler calls the network through

    fn(x, t, kappa) -> logits

where x is a state, t [B] is its nominal time (the trainer and the sampler always know it), and kappa [B] is the
fraction of the way to the denoised point covered by the jump: Phi(x) = (1 - kappa) x + kappa softmax(fn(x, t, kappa)),
which lands at nominal time t' = 1 - (1 - kappa)(1 - t). kappa = 0 is the denoiser.

    cond          inputs of the network                               time-free?
    duration      sigma = clock.kappa_inv(kappa)                      yes: the equilibrium flow map of the notes
    state_time    tau(t_hat(x)) and tau(t'), t_hat from the norm of x  yes: a function of (x, kappa) only
    time          tau(t) and tau(t')                                  no: FLM (kappa = 0) and FMLM

A field (`second_input: false`) only receives the first input, and kappa must be 0.
"""
import torch

from .reparam import get_reparam, state_time


def make_fn(net, cond, clock, vocab_size, **kw):
    """Returns fn(x, t, kappa) -> logits for `net`, a Denoiser or a DiT (possibly wrapped for several GPUs)."""
    two = getattr(getattr(net, "module", net), "sigma_map_prime", None) is not None

    if cond == "duration":
        def fn(x, t, kappa):
            return net(x, clock.kappa_inv(kappa), **kw)
        fn.in_sigma = lambda x, sigma: net(x, sigma, **kw)   # for derivatives with respect to the duration label
        return fn

    if cond not in ("state_time", "time"):
        raise ValueError(f"unknown model.cond '{cond}'")
    rep = get_reparam(vocab_size)

    def fn(x, t, kappa):
        if cond == "state_time":
            t = state_time(x)
        if not two:
            return net(x, rep.tau(t))
        return net(x, rep.tau(t), rep.tau(1 - (1 - kappa) * (1 - t)))
    return fn


def probs_fn(fn):
    def p(x, t, kappa):
        return fn(x, t, kappa).float().softmax(dim=-1)
    if hasattr(fn, "in_sigma"):
        p.in_sigma = lambda x, sigma: fn.in_sigma(x, sigma).float().softmax(dim=-1)
    return p


def flatten_first_input(model, at=0.0):
    """Make a DiT ignore its first input, keeping the output it has when that input equals `at`.

    Used to start a duration-conditioned flow map with delta_sigma = D for every sigma, and to start a time-free
    network from a time-conditioned one (frozen at the time `at`)."""
    with torch.no_grad():
        value = model.sigma_map(torch.full((1,), float(at), device=model.sigma_map.mlp[2].bias.device))
        model.sigma_map.mlp[2].weight.zero_()
        model.sigma_map.mlp[2].bias.copy_(value[0])


def init_from(model, cfg, kind):
    """`train.init`: start from the EMA weights of another checkpoint (or `released:<dir>`)."""
    from .trainer import load_model
    src, ckpt = load_model(cfg.train.init)
    src_cond, cond = ckpt["cfg"]["model"]["cond"], cfg.model.cond
    missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("sigma_map_prime") for k in missing), (missing, unexpected)
    if cfg.model.arch == "dit" and cond == "duration":
        if src_cond != "duration":
            flatten_first_input(model, at=cfg.train.init_time)   # time-conditioned -> time-free
        elif kind == "flow_map" and ckpt["kind"] == "field" and cfg.loss.flat_init:
            flatten_first_input(model, at=0.0)
    elif (cond == "duration") != (src_cond == "duration"):
        raise ValueError(f"cannot start a '{cond}' model from a '{src_cond}' checkpoint")
