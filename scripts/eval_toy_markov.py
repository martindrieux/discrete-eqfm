"""Samples a checkpoint trained on the Markov toy (data.name = markov) with every sampler and prints exact metrics.

    python scripts/eval_toy_markov.py CKPT [out=results.json] [num=4000] [h=0.5,0.7,0.9]

Samplers, each with K network evaluations:
    plain          Algorithm 5 of the notes: K jumps of the flow map on a uniform time grid (a denoiser: K Euler steps)
    adaptive       the same grid, positions committed when the jump's own output is confident (Section 7.2)
    adaptive-pre   positions committed from the map to the end before each jump (two evaluations per step)
    refine         DBTM: the map to the end, commit the confident positions, fresh noise on the others
    refine-keep    the same, the other positions keep their noise
    speed          a denoiser with one time per position, advancing at the speed 1 + beta * confidence (threshold column: beta)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deqfm.clocks import build_clock
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample_adaptive, sample_adaptive_speed, sample_grid, sample_refine, time_grid


def main():
    path = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    num = int(opt.get("num", 4000))
    hs = [float(v) for v in opt.get("h", "0.5,0.7,0.9").split(",")]
    gk = opt.get("grid", "uniform")          # uniform | reparam (uniform in the decoding-error clock of this vocabulary)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    data = build_dataset(cfg["data"])
    model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"]).eval().to(dev)
    model.load_state_dict(ckpt["ema"])
    clock, cond = build_clock(cfg["clock"]), cfg["model"]["cond"]
    fm = ckpt["kind"] == "flow_map"
    rows = []

    def add(name, K, tokens, evals=None, h=None):
        rows.append({"sampler": name, "K": K, "evals": evals or K, "h": h, **data.metrics(tokens)})
        r = rows[-1]
        print(f"| {name} | {r['h'] if h is not None else ''} | {r['evals']} | {r['valid']:.3f} | {r['step_valid']:.4f} | "
              f"{r['nll']:.3f} | {r['pair_tv']:.3f} | {r['distinct']:.3f} |", flush=True)

    real = data.metrics(data.draw(num))
    print(f"checkpoint {path} step {ckpt['step']} kind {ckpt['kind']}; entropy rate {data.entropy_rate:.3f}")
    print(f"real data: valid {real['valid']:.3f} nll {real['nll']:.3f} pair_tv {real['pair_tv']:.3f} (sampling noise floor)")
    print("| sampler | threshold | evaluations | valid | allowed transitions | nll | pair TV | distinct |")
    print("|---|---|---|---|---|---|---|---|")
    Ks = (1, 2, 4, 8, 16) if fm else (2, 4, 8, 16, 64, 256)
    if "K" in opt:
        Ks = [int(v) for v in opt["K"].split(",")]
    for K in Ks:
        torch.manual_seed(0)
        grid = time_grid(K, gk, ckpt["vocab_size"])
        add("plain", K, sample_grid(model, cond, clock, num, grid, flow_map=fm, device=dev))
        if K == 1:
            continue
        for h in hs:
            torch.manual_seed(0)
            tok, ev = sample_adaptive(model, cond, clock, num, grid, h=h, when="post", flow_map=fm, device=dev)
            add("adaptive", K, tok, ev, h)
        if K % 2 == 0:      # K evaluations in total: K / 2 + 1 jumps would overshoot, so use K / 2 jumps (K - 1 evaluations)
            for h in hs:
                torch.manual_seed(0)
                tok, ev = sample_adaptive(model, cond, clock, num, time_grid(K // 2, gk, ckpt["vocab_size"]), h=h, when="pre", flow_map=fm, device=dev)
                add("adaptive-pre", K, tok, ev, h)
        if not fm:      # one time per position: the threshold column holds beta
            for beta in [float(v) for v in opt.get("beta", "1,4,16").split(",")]:
                torch.manual_seed(0)
                add("speed", K, sample_adaptive_speed(model, cond, clock, num, grid, beta=beta, device=dev), K, beta)
        if fm:
            for h in hs:
                for fresh in (True, False):
                    torch.manual_seed(0)
                    add("refine" if fresh else "refine-keep", K,
                        sample_refine(model, cond, clock, num, K, threshold=h, device=dev, fresh=fresh), K, h)
    if "out" in opt:
        with open(opt["out"], "w") as fjs:
            json.dump({"ckpt": path, "step": ckpt["step"], "real": real, "entropy_rate": data.entropy_rate, "rows": rows}, fjs, indent=1)


if __name__ == "__main__":
    main()
