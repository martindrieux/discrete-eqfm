"""Mean and spread over seeds of the key toy variants (60k steps, decoding-clock grid).   python scripts/toy_seeds.py
Cells: valid sequences % (mean ± spread over seeds) / pair TV; the best threshold of each sampler per run."""
import json
import os
import statistics as st

G = {"Denoiser, no clean positions": ["TM4_long_denoiser_plain", "TM5_denoiser_plain_s1", "TM5_denoiser_plain_s2"],
     "Denoiser, random clean positions": ["TM4_long_denoiser_context_half", "TM5_denoiser_context_half_s1", "TM5_denoiser_context_half_s2"],
     "Denoiser, model-chosen clean positions (h = 0.9)": ["TM4_long_denoiser_adapt_h90_half", "TM5_denoiser_adapt_h90_half_s1", "TM5_denoiser_adapt_h90_half_s2"],
     "Flow map, no clean positions": ["TM4_long_plain", "TM5_plain_s1", "TM5_plain_s2"],
     "Flow map, random clean positions": ["TM4_long_context_half", "TM5_context_half_s1", "TM5_context_half_s2"],
     "Flow map, model-chosen clean positions (h = 0.9, half the samples)": ["TM4_long_adapt_h90_half", "TM5_adapt_h90_half_s1", "TM5_adapt_h90_half_s2"],
     "Flow map, model-chosen clean positions (h = 0.9, all samples)": ["TM4_long_adapt_h90", "TM5_adapt_h90_s1", "TM5_adapt_h90_s2"],
     "Flow map, model-chosen clean positions (h = 0.7, half the samples)": ["TM4_long_adapt_h70_half", "TM5_adapt_h70_half_s1", "TM5_adapt_h70_half_s2"]}
S = [("plain", "plain steps"), ("adaptive", "steps with commits"), ("refine-keep", "map to the end with commits")]


def best(rows, K, s):
    r = [x for x in rows if x["K"] == K and x["sampler"] == s]
    return max(r, key=lambda x: x["valid"] - x["pair_tv"]) if r else None


for kind, Ks in (("Denoiser", (8, 16, 64, 256)), ("Flow map", (1, 2, 4, 8, 16))):
    print(f"\n| {kind} trained on | seeds | sampler | " + " | ".join(f"{K} evaluations" for K in Ks) + " |")
    print("|---|---|---|" + "---|" * len(Ks))
    for g, names in G.items():
        if not g.startswith(kind):
            continue
        R = [json.load(open(f"runs/{n}/toy_eval.json"))["rows"] for n in names if os.path.exists(f"runs/{n}/toy_eval.json")]
        first = True
        for s, label in S:
            cells = []
            for K in Ks:
                v = [b for b in (best(r, K, s) for r in R) if b]
                if not v:
                    cells.append("")
                    continue
                va = [100 * x["valid"] for x in v]
                cells.append(f"{st.mean(va):.0f} ± {st.pstdev(va):.0f} / {st.mean(x['pair_tv'] for x in v):.3f}")
            if any(cells):
                print(f"| {g.split(', ', 1)[1] if first else ''} | {len(R) if first else ''} | {label} | " + " | ".join(cells) + " |")
                first = False
