"""Sample every checkpoint under a directory of toy runs and print a table of validity and mode coverage.

    python scripts/eval_toy.py runs/toy_validation [num_samples]

valid : fraction of samples that are one of the data sequences (1 is perfect)
TV    : total variation distance between the sampled and the true sequence frequencies (0 is perfect)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deqfm.clocks import build_clock
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample


def main():
    root = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    torch.manual_seed(0)
    print("| run | kind | steps trained | network evaluations | valid | TV |")
    print("|---|---|---|---|---|---|")
    for run in sorted(os.listdir(root)):
        path = os.path.join(root, run, "last.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        dataset = build_dataset(cfg["data"])
        model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"]).eval()
        model.load_state_dict(ckpt["ema"])
        flow_map = ckpt["kind"] == "flow_map"
        for steps in ((1, 2, 4, 8) if flow_map else (1, 2, 4, 8, 16, 64)):
            tokens = sample(model, build_clock(cfg["clock"]), num, steps, flow_map=flow_map)
            print(f"| {run} | {ckpt['kind']} | {ckpt['step']} | {steps} | "
                  f"{dataset.valid_rate(tokens):.3f} | {dataset.mode_tv(tokens):.3f} |", flush=True)


if __name__ == "__main__":
    main()
