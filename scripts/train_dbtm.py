"""Train a Discrete Beckmann Transport Model (baseline): one time-free map T(x) to the data. See deqfm/dbtm.py.

    python scripts/train_dbtm.py configs/wt2.yaml out_dir=runs/dbtm [dbtm.ril=true dbtm.ril_start=15000]

The checkpoint is a flow map whose only trained jump is the whole one (kappa = 1): sample it with one step, or with the
refinement sampler (scripts/eval_lm.py ... sampler=refine).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deqfm.config import parse_cli
from deqfm.data import build_dataset
from deqfm.dbtm import make_dbtm_loss
from deqfm.model import build_model
from deqfm.trainer import train


def main():
    cfg = parse_cli()
    assert cfg.model.arch == "dit" and cfg.model.cond == "duration", "DBTM uses the time-free DiT"
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size, dataset.seq_len, "flow_map")
    train(cfg, model, dataset, make_dbtm_loss(cfg, dataset.vocab_size), kind="flow_map")


if __name__ == "__main__":
    main()
