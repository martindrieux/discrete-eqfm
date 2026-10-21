"""WikiText-2: flow map with one duration per position against plain EqFM, same teacher and steps, mean ± spread over seeds.
python scripts/speedmap_table.py"""
import glob, json, statistics as st

GROUPS = {"Plain EqFM (one duration for the sequence)": "W2W_plain_s", "Adaptive flow map, semigroup target": "W2W_speed_s",
          "Adaptive flow map, Lagrangian target": "W2W_speedlag_s",
          "Version 2: plain EqFM, Lagrangian target": "W2X_lagplain_s", "Version 2: adaptive flow map, Lagrangian target": "W2X_lagspeed_s",
          "Version 3: adaptive flow map, semigroup target, finishing positions from the teacher": "W2Y_speedsg_s"}


def cell(vals):
    if not vals: return ""
    p, e, d = ([v[k] for v in vals] for k in ("gen_ppl", "entropy", "dup1"))
    sd = f" ± {st.pstdev(p):.0f}" if len(p) > 1 else ""
    return f"{st.mean(p):.0f}{sd} / {st.mean(e):.2f} / {100 * st.mean(d):.1f}% ({len(p)})"


for K in ("4", "8", "16"):
    print(f"\n| {K} evaluations: perplexity / entropy / doubled tokens (seeds) | plain sampler | " + " | ".join(f"speed 1 + {b}g" for b in (1, 2, 4, 8, 16)) + " |")
    print("|---|" + "---|" * 6)
    for name, pre in GROUPS.items():
        row = []
        for b in (0, 1, 2, 4, 8, 16):
            vals = [json.load(open(f))["steps"][K] for f in sorted(glob.glob(f"runs/{pre}*/eval_speedmap_beta{b}.json"))]
            row.append(cell(vals))
        print(f"| {name} | " + " | ".join(row) + " |")
print("\n| Ordinary evaluation (uniform grid): perplexity / entropy | 1 | 2 | 4 | 8 |\n|---|---|---|---|---|")
for name, pre in GROUPS.items():
    ev = [json.load(open(f))["steps"] for f in sorted(glob.glob(f"runs/{pre}*/eval.json"))]
    print(f"| {name} | " + " | ".join(cell([e[K] for e in ev]) for K in ("1", "2", "4", "8")) + " |")
