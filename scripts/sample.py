"""Sample from a checkpoint and print decoded sequences.

    python scripts/sample.py runs/toy_eul/last.pt sample.steps=4 sample.num=8

A field checkpoint is sampled by integrating dx/dtau = D(x) - x; a flow-map checkpoint with Algorithm 5.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deqfm.clocks import build_clock
from deqfm.config import load_config
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample, unigram_entropy


def main():
    path, overrides = sys.argv[1], sys.argv[2:]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config(None, overrides)          # defaults + command-line overrides (sample.*)
    for key in ("data", "model", "clock"):      # architecture, data and clock come from the checkpoint
        cfg[key] = ckpt["cfg"][key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(cfg["data"])
    model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"]).to(device).eval()
    model.load_state_dict(ckpt["ema"])
    tokens = sample(model, build_clock(cfg["clock"]), cfg.sample.num, cfg.sample.steps,
                    flow_map=ckpt["kind"] == "flow_map", device=device, cond=cfg["model"].get("cond", "duration"))
    for row in tokens:
        print(dataset.decode(row.cpu()))
    print(f"\n{ckpt['kind']} | step {ckpt['step']} | {cfg.sample.steps} network evaluations | "
          f"unigram entropy {unigram_entropy(tokens.cpu()):.3f}")
    if hasattr(dataset, "valid_rate"):
        print(f"valid rate {dataset.valid_rate(tokens):.3f} | mode TV {dataset.mode_tv(tokens):.3f}")


if __name__ == "__main__":
    main()
