"""Minimal YAML config with defaults, attribute access and `a.b=value` command-line overrides."""
import copy
import sys

import yaml

DEFAULTS = {
    "seed": 0,
    "out_dir": "runs/default",
    "clock": "exponential",              # exponential | rational
    "data": {
        "name": "toy",                   # toy | text | tokens | blocks
        "seq_len": 16,
        # toy
        "vocab_size": 16, "n_modes": 8,
        # text (character level), tokens (pre-tokenised memmap) and blocks (pre-cut sequences, [N, seq_len])
        "path": None,
    },
    "model": {"arch": "denoiser",        # denoiser (small transformer, absolute positions) | dit (the FLM backbone)
              "dim": 256, "depth": 4, "heads": 4, "mlp_ratio": 4, "dropout": 0.0,
              "sigma_scale": 1.0,        # denoiser: largest frequency of the duration embedding (radians per unit sigma)
              "cond": "duration",        # duration | state_time | time: what the network is told (deqfm/nets.py)
              "cond_dim": 128, "softcap": 50.0, "logit_cap": 30.0},   # dit only
    "loss": {
        # diagonal term: t ~ q, weight 1 / ((1 - t) q(t))
        "diag_time": "uniform",          # uniform: q = U[0, 1 - t_eps] | log_uniform: q prop. to 1/(1-t), constant weight
                                         # | reparam: uniform in the decoding-error clock of FLM, weight 1
        "stratified": False,             # one time sample per equal-probability bin of the batch
        "t_eps": 1.0e-4,                 # numerical guard on t near 1
        # flow map only
        "off_diagonal": "semigroup",     # semigroup | lagrangian | eulerian
        "alpha": 1.0, "diag_weight": 1.0,   # loss = diag_weight * diag + alpha * off (diag_weight 0: no denoiser is trained)
        "direction": "denoiser",         # eulerian: derivative along Dbar(x) - x (denoiser) or along x_1 - x (data)
        "jvp": "forward",                # forward (torch.func.jvp) | fd (central finite differences)
        "fd_eps": 1.0e-3,
        "split_batch": True,             # half of each batch per term (as in FMLM); else both terms on the full batch
        "teacher": None,                 # path to a field checkpoint: distillation. None: training from data
        "teacher_cond": None,            # how the teacher is called; default: as it was trained
        "init_from_teacher": True,
        "flat_init": True,               # dit, cond = duration: start with delta_sigma = teacher denoiser for every sigma
        "jump_sampling": "notes",        # notes: t and sigma uniform | reparam: as FMLM, uniform in the decoding-error clock
                                         # | endpoint: only the whole jump kappa = 1
        "whole_jump_prob": 0.0,          # reparam: probability of the whole jump from noise (FMLM uses 1/32)
        # Discrete Beckmann Transport Models: partial context and refinement in the loop (deqfm/losses.py)
        "partial_context": False,        # per sample, a fraction f ~ U[0, 1] of the positions is clean
        "ril_frac": 0.0,                 # fraction of the batch rolled out through the refinement rounds (0: off)
        "ril_commit": "data",            # data: commit data tokens by the confidence on them (DBTM training) | model: commit
                                         # the model's own argmax tokens by their confidence (as at sampling time)
        "ril_distill_target": "soft",    # soft: a frozen copy's probabilities at the last round | sample: the tokens the
                                         # rollout ends with (needs ril_commit = model)
        "ril_start": 0,                  # training step at which the refinement losses are switched on
        "ril_depth": 4, "ril_threshold": 0.9, "ril_weight": 1.0, "ril_ot": True,
        "ril_adaptive_r": 0.0, "ril_adaptive_c": 1.0,   # each sequence's refinement loss is weighted by (loss + c)^(-r)
        "ril_distill_weight": 0.0, "ril_distill_start": 0, "ril_distill_temp": 0.7, "ril_refresh": 5000,
        "target_net": "online",          # online | ema: weights used to build the off-diagonal targets
        "target_eval": False,            # build the targets without dropout
        "end_jump_prob": 0.0,            # reparam: probability of a jump to the end (kappa = 1) from a state at a uniform tau
        "whole_jump_target": "same",     # same | semigroup: target used for kappa = 1 under the Lagrangian / Eulerian loss
        "boundary_prob": 0.0,            # fraction of off-diagonal samples used for the boundary term (see objectives.py)
        "boundary_t": 0.9, "boundary_target": "denoiser",   # denoiser | data
    },
    "train": {
        "steps": 100000, "batch_size": 64, "grad_accum": 1,
        "lr": 3.0e-4, "warmup": 2500, "lr_decay": "none", "betas": [0.9, 0.999], "weight_decay": 0.0,
        "grad_clip": 1.0, "ema": 0.9999,
        "precision": "no",               # no | fp16 | bf16   (passed to accelerate)
        "num_workers": 4, "log_every": 100, "save_every": 10000,
        "resume": True,                  # continue from out_dir/last.pt if it exists
        "init": None,                    # start from the EMA weights of this checkpoint (or released:<dir>)
        "init_time": 0.5,                # reparameterised time at which a time-conditioned init is frozen (cond = duration)
        "min_rate": 0,                   # > 0: stop (as if interrupted) if the speed after 400 steps is below this many it/s
        "max_minutes": 0,                # > 0: checkpoint and exit with code 3 after this many minutes (job time limits)
        "keep_every": 0,                 # > 0: also keep step_XXXXXXX.pt (EMA weights only) every this many steps
    },
    "ckpt_dir": None,                    # where checkpoints go (default: out_dir); logs and results stay in out_dir
    "sample": {"steps": 8, "num": 16},
    "eval": {"steps": [1, 2, 4, 8], "num": 1024, "batch": 128, "grid": "reparam",   # grid: reparam | uniform
             "judge": "gpt2-large", "at_end": False},
    "wandb": {"project": None, "name": None},
}


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    __setattr__ = dict.__setitem__


def _merge(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _wrap(d):
    return Cfg({k: _wrap(v) if isinstance(v, dict) else v for k, v in d.items()})


def _number(v):
    """YAML reads 1e-4 (no decimal point) as a string."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _number(yaml.safe_load(value))
    return _wrap(cfg)


def parse_cli(argv=None):
    """usage: script.py [config.yaml] [key=value ...]"""
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv and "=" not in argv[0] else None
    return load_config(path, argv[1:] if path else argv)


def to_dict(cfg):
    return {k: to_dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
