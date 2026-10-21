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
        "branching": 4,                  # markov: allowed successors per token
        # text (character level), tokens (pre-tokenised memmap) and blocks (pre-cut sequences, [N, seq_len])
        "path": None,
    },
    "model": {"arch": "denoiser",        # denoiser (small transformer, absolute positions) | dit (the FLM backbone)
              "dim": 256, "depth": 4, "heads": 4, "mlp_ratio": 4, "dropout": 0.0,
              "sigma_scale": 1.0,        # denoiser: largest frequency of the duration embedding (radians per unit sigma)
              "cond": "duration",        # duration | state_time | time: what the network is told (deqfm/nets.py)
              "cond_dim": 128, "softcap": 50.0, "logit_cap": 30.0,    # dit only
              "self_cond": False},       # dit: a second input, the draft p of the clean tokens (ELF self-conditioning)
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
        "random_speed": 0.0,             # denoiser: beta_max > 0 gives one time per position, t^l = min(1, t (1 + beta g^l)),
                                         # g^l ~ U[0, 1], beta ~ U[0, beta_max], on a share context_frac of the samples
        "context_frac": 1.0,             # fraction of the samples that get clean positions
        "ril_frac": 0.0,                 # fraction of the batch rolled out through the refinement rounds (0: off)
        "ril_commit": "data",            # data: commit data tokens by the confidence on them (DBTM training) | model: commit
                                         # the model's own argmax tokens by their confidence (as at sampling time)
        "ril_distill_target": "soft",    # soft: a frozen copy's probabilities at the last round | sample: the tokens the
                                         # rollout ends with (needs ril_commit = model)
        "ril_rollout_net": "online",     # online: the trained network plays the rollout | snapshot: the frozen copy does
        "ril_start": 0,                  # training step at which the refinement losses are switched on
        "ril_depth": 4, "ril_threshold": 0.9, "ril_weight": 1.0, "ril_ot": True,
        "ril_adaptive_r": 0.0, "ril_adaptive_c": 1.0,   # each sequence's refinement loss is weighted by (loss + c)^(-r)
        "ril_distill_weight": 0.0, "ril_distill_start": 0, "ril_distill_temp": 0.7, "ril_refresh": 5000,
        "target_net": "online",          # online | ema: weights used to build the off-diagonal targets
        "target_eval": False,            # build the targets without dropout
        "end_jump_prob": 0.0,            # reparam: probability of a jump to the end (kappa = 1) from a state at a uniform tau
        "whole_jump_target": "same",     # same | semigroup: target used for kappa = 1 under the Lagrangian / Eulerian loss
        # Section 7 of the notes: states with clean ("committed") positions chosen by the model's own confidence
        "adaptive": "off",               # off | binary: positions whose confidence exceeds adaptive_h are clean (t = 1)
                                         # | rollout: as binary, but committed step by step along the time grid, as the sampler does
                                         # | speed: one time per position, advancing at the speed 1 + adaptive_beta * confidence
                                         # | speed_clock: progress in the decoding-error clock at the speed base + beta * confidence^power
        "adaptive_beta": 4.0,            # speed: psi(g) = 1 + beta g
        "adaptive_power": 8.0, "adaptive_base": 1.0,   # speed_clock: psi(g) = base + beta g^power
        "adaptive_h": 0.9,               # commit threshold on the confidence score
        "adaptive_score": "maxprob",     # maxprob | margin (largest minus second largest) | certainty (1 - entropy / log V)
        "adaptive_start": 0,             # warmup: training step from which the adaptive states are used
        "adaptive_frac": 1.0,            # fraction of the samples that get adaptive states (the others are plain I_t)
        "adaptive_rollout_steps": 4,     # rollout: number of grid steps between time 0 and the state's time
        "position_speed": 0.0,           # > 0: flow map with one duration per position (losses.position_speed_state / _jump);
        "position_speed_short": 0.5,     # share of the jumps drawn as a step inside the decoding clock, tau(t') ~ U[tau(t), 1]
        "position_speed_finish": "self", # semigroup target of a position that finishes inside a jump: self | teacher
        "position_speed_power": 1.0,     # the value is beta_max in the speed law psi(g) = 1 + beta g^power, beta ~ U[0, beta_max]
        "freeze_context": False,         # clean positions do not move in the jumps that build the targets, and the
                                         # losses are taken on the other positions only (equation 30 of the notes)
        "boundary_prob": 0.0,            # fraction of off-diagonal samples used for the boundary term (see objectives.py)
        "boundary_t": 0.9, "boundary_target": "denoiser",   # denoiser | data
        # self-conditioning (model.self_cond): the draft is the network's own denoiser output without draft, stop-gradient,
        "self_cond_prob": 0.5,           # on this share of the rows (the others get no draft)
        "self_cond_view": "same",        # same: computed from the same state | other: from an independent noisy view of the
                                         # same sequence (new noise, own time and clean positions), as after a renoise
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
        "val_every": 0,                  # > 0: held-out denoising loss every this many steps (needs eval.val_path); the
                                         # averaged weights with the lowest value are kept as best.pt
        "min_rate": 0,                   # > 0: stop (as if interrupted) if the speed after 400 steps is below this many it/s
        "max_minutes": 0,                # > 0: checkpoint and exit with code 3 after this many minutes (job time limits)
        "keep_every": 0,                 # > 0: also keep step_XXXXXXX.pt (EMA weights only) every this many steps
    },
    "ckpt_dir": None,                    # where checkpoints go (default: out_dir); logs and results stay in out_dir
    "dbtm": {                            # Discrete Beckmann Transport Models baseline (deqfm/dbtm.py, scripts/train_dbtm.py)
        "ot": True, "partial_context": True, "t_anchor": 0.75, "lambda_anchor": 1.0, "lambda_bnd": 1.0,
        "adaptive_r": 0.5, "adaptive_c": 1.0,
        "ril": False, "ril_start": 0, "ril_frac": 1.0, "ril_depth": 4, "ril_threshold": 0.9,
        "ril_distill_temp": 0.7, "ril_refresh": 5000, "lambda_ril": 1.0, "lambda_semi": 1.0,
    },
    "sample": {"steps": 8, "num": 16},
    "sdist": {"K": 32, "beta": 2.0, "power": 1.0, "base": 1.0},   # scripts/train_sample_distill.py: the teacher's sampler
    "eval": {"steps": [1, 2, 4, 8], "num": 1024, "batch": 128, "grid": "reparam",   # grid: reparam | uniform
             "judge": "gpt2-large", "at_end": False,
             "val_path": None, "val_num": 1024},   # held-out sequences for train.val_every
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
