"""Train the equilibrium flow map delta_hat_sigma (Algorithm 1 with Algorithm 2, 3 or 4).

From data:
    python scripts/train_flow_map.py configs/toy.yaml loss.off_diagonal=eulerian out_dir=runs/toy_eul
Distillation from a trained field:
    python scripts/train_flow_map.py configs/toy.yaml loss.teacher=runs/toy_field/last.pt out_dir=runs/toy_distill
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deqfm.clocks import build_clock
from deqfm.config import parse_cli
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.nets import flatten_first_input, init_from
from deqfm.objectives import make_flow_map_loss, make_val_loss
from deqfm.trainer import load_model, train


COMPATIBLE = {"duration": ("duration",), "state_time": ("state_time", "time", "duration"), "time": ("time", "state_time")}


def flatten_duration(model):
    """Make a duration-conditioned DiT ignore sigma, keeping its output at sigma = 0: delta_sigma = D for every sigma."""
    flatten_first_input(model, at=0.0)


def main():
    cfg = parse_cli()
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size, dataset.seq_len, "flow_map")
    teacher, teacher_cond = None, None
    if cfg.loss.teacher:
        teacher, ckpt = load_model(cfg.loss.teacher)
        assert ckpt["kind"] == "field", "the teacher must be a field checkpoint"
        trained_cond = ckpt["cfg"]["model"]["cond"]
        teacher_cond = cfg.loss.teacher_cond or trained_cond
        if cfg.loss.init_from_teacher and not cfg.train.init:
            if trained_cond not in COMPATIBLE[cfg.model.cond]:
                raise ValueError(f"cannot start a '{cfg.model.cond}' model from a '{trained_cond}' teacher; "
                                 "set loss.init_from_teacher=false or train.init")
            model.load_state_dict(teacher.state_dict(), strict=False)   # a second time input, if any, starts at zero
            if cfg.model.arch == "dit" and cfg.model.cond == "duration" and cfg.loss.flat_init:
                flatten_duration(model)
            if cfg.model.cond == "state_time" and trained_cond == "duration":
                flatten_duration(model)   # the teacher never used its input: start equal to it for every state time
    if cfg.train.init:      # takes precedence over the teacher's weights
        init_from(model, cfg, "flow_map")
    clock = build_clock(cfg.clock)
    loss_fn = make_flow_map_loss(cfg, dataset.vocab_size, clock, teacher, teacher_cond)
    train(cfg, model, dataset, loss_fn, kind="flow_map", teacher=teacher, val_fn=make_val_loss(cfg, dataset.vocab_size, clock))


if __name__ == "__main__":
    main()
