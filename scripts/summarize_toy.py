"""One table for the Markov toy: per run and per number of network evaluations, the plain sampler and the best
threshold of every other sampler.    python scripts/summarize_toy.py runs/TM_*   [metric=valid]
Cells: valid sequences % / pair TV (lower is better; real data ~0.02) / mean NLL of allowed transitions."""
import json
import os
import sys

runs = [a for a in sys.argv[1:] if "=" not in a]
FILE = dict(a.split("=") for a in sys.argv[1:] if "=" in a).get("file", "toy_eval.json")
samplers = ["plain", "adaptive", "adaptive-pre", "speed", "refine", "refine-keep"]
for run in runs:
    path = os.path.join(run, FILE)
    if not os.path.exists(path):
        continue
    r = json.load(open(path))
    print(f"\n### {os.path.basename(run)}  (step {r['step']}; real data: nll {r['real']['nll']:.3f}, pair TV {r['real']['pair_tv']:.3f})\n")
    print("| evaluations | " + " | ".join(samplers) + " |")
    print("|---|" + "---|" * len(samplers))
    for K in sorted({x["K"] for x in r["rows"]}):
        cells = []
        for s in samplers:
            rows = [x for x in r["rows"] if x["K"] == K and x["sampler"] == s]
            if not rows:
                cells.append("")
                continue
            b = max(rows, key=lambda x: x["valid"] - x["pair_tv"])      # best threshold: valid and varied
            h = f" (h={b['h']})" if b["h"] is not None else ""
            cells.append(f"{100 * b['valid']:.1f}% / {b['pair_tv']:.3f} / {b['nll']:.2f}{h}")
        print(f"| {K} | " + " | ".join(cells) + " |")
