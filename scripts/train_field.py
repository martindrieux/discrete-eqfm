"""Train the autonomous field only: a denoiser D_hat with no time input, by weighted cross-entropy.

    python scripts/train_field.py configs/toy.yaml out_dir=runs/toy_field
    accelerate launch scripts/train_field.py configs/owt.yaml out_dir=runs/owt_field
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deqfm.clocks import build_clock
from deqfm.config import parse_cli
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.nets import init_from
from deqfm.objectives import make_field_loss, make_val_loss
from deqfm.trainer import train


def main():
    cfg = parse_cli()
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size, dataset.seq_len, "field")
    if cfg.train.init:
        init_from(model, cfg, "field")
    clock = build_clock(cfg.clock)
    train(cfg, model, dataset, make_field_loss(cfg, dataset.vocab_size, clock), kind="field",
          val_fn=make_val_loss(cfg, dataset.vocab_size, clock))


if __name__ == "__main__":
    main()
