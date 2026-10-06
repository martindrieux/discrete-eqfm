"""Minimal YAML config with defaults, attribute access and `a.b=value` command-line overrides."""
import copy
import sys

import yaml

DEFAULTS = {
    "seed": 0,
    "out_dir": "runs/default",
    "clock": "exponential",              # exponential | rational
    "data": {
        "name": "toy",                   # toy | text | tokens
        "seq_len": 16,
        # toy
        "vocab_size": 16, "n_modes": 8,
        # text (character level) and tokens (pre-tokenised memmap)
        "path": None,
    },
    "model": {"dim": 256, "depth": 4, "heads": 4, "mlp_ratio": 4, "dropout": 0.0,
              "sigma_scale": 100.0},     # largest frequency of the duration embedding (radians per unit sigma)
    "loss": {
        # diagonal term: t ~ q, weight 1 / ((1 - t) q(t))
        "diag_time": "uniform",          # uniform: q = U[0, 1 - t_eps] | log_uniform: q prop. to 1/(1-t), constant weight
        "t_eps": 1.0e-4,                 # numerical guard on t near 1
        # flow map only
        "off_diagonal": "semigroup",     # semigroup | lagrangian | eulerian
        "alpha": 1.0,                    # loss = diag + alpha * off
        "jvp": "forward",                # forward (torch.func.jvp) | fd (central finite differences)
        "fd_eps": 1.0e-3,
        "split_batch": True,             # half of each batch per term (as in FMLM); else both terms on the full batch
        "teacher": None,                 # path to a field checkpoint: distillation. None: training from data
        "init_from_teacher": True,
    },
    "train": {
        "steps": 100000, "batch_size": 64, "grad_accum": 1,
        "lr": 3.0e-4, "warmup": 2500, "betas": [0.9, 0.999], "weight_decay": 0.0,
        "grad_clip": 1.0, "ema": 0.9999,
        "precision": "no",               # no | fp16 | bf16   (passed to accelerate)
        "num_workers": 4, "log_every": 100, "save_every": 10000,
        "resume": True,                  # continue from out_dir/last.pt if it exists
    },
    "sample": {"steps": 8, "num": 16},
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
        node[leaf] = yaml.safe_load(value)
    return _wrap(cfg)


def parse_cli(argv=None):
    """usage: script.py [config.yaml] [key=value ...]"""
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv and "=" not in argv[0] else None
    return load_config(path, argv[1:] if path else argv)


def to_dict(cfg):
    return {k: to_dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
