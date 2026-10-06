"""Train the equilibrium flow map delta_hat_sigma (Algorithm 1 with Algorithm 2, 3 or 4).

From data:
    python scripts/train_flow_map.py configs/toy.yaml loss.off_diagonal=eulerian out_dir=runs/toy_eul
Distillation from a trained field:
    python scripts/train_flow_map.py configs/toy.yaml loss.teacher=runs/toy_field/last.pt out_dir=runs/toy_distill
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deqfm.clocks import build_clock
from deqfm.config import parse_cli
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.objectives import make_flow_map_loss
from deqfm.trainer import load_checkpoint, train


def main():
    cfg = parse_cli()
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size, dataset.seq_len)
    teacher = None
    if cfg.loss.teacher:
        teacher = build_model(cfg.model, dataset.vocab_size, dataset.seq_len)
        load_checkpoint(cfg.loss.teacher, teacher, use_ema=True)
        if cfg.loss.init_from_teacher:
            model.load_state_dict(teacher.state_dict())
    loss_fn = make_flow_map_loss(cfg, dataset.vocab_size, build_clock(cfg.clock), teacher)
    train(cfg, model, dataset, loss_fn, kind="flow_map", teacher=teacher)


if __name__ == "__main__":
    main()
