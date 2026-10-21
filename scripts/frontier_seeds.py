"""Equal-entropy comparison on WikiText-2 over seeds.   python scripts/frontier_seeds.py
For each run, the noise-scale sweep gives a curve (entropy, perplexity); the perplexity at a target entropy is read off
by linear interpolation. Cells: mean ± spread over seeds, or the entropy range the model can reach when the target is outside it."""
import glob, json, os, statistics as st

G = {"Beckmann, plain": ["W2_dbtm", "W2S_dbtm_s1", "W2S_dbtm_s2"],
     "Beckmann, refinement in the loop": ["W2_dbtm_ril", "W2S_dbtm_ril_s1", "W2S_dbtm_ril_s2"],
     "EqFM from data (no teacher)": ["W2C_eqfm_data_semigroup_wholejump", "W2S_eqfm_data_s1", "W2S_eqfm_data_s2"],
     "EqFM in two stages from data (3000 + 3000 steps, no outside teacher)": ["W2T_twostage_s0", "W2T_twostage_s1", "W2T_twostage_s2"],
     "One-step map trained on the denoiser's continuous-speed samples": ["W2V_onestep_s0", "W2V_onestep_s1", "W2V_onestep_s2", "W2V_onestep_s3"],
     "EqFM in two stages from data, 2000 + 4000 steps": ["W2U_long2_s0", "W2U_long2_s1", "W2U_long2_s2"],
     "EqFM in two stages from data, forced whole jump 1 in 8": ["W2U_whole8_s0", "W2U_whole8_s1", "W2U_whole8_s2"],
     "EqFM in two stages from data, Eulerian target": ["W2U_eul_s0", "W2U_eul_s1", "W2U_eul_s2"],
     "EqFM from data, Lagrangian, targets without dropout": ["W2T_lag_nodrop_s0", "W2T_lag_nodrop_s1", "W2T_lag_nodrop_s2"],
     "EqFM from data, Lagrangian": ["W2C_eqfm_data_lagrangian", "W2T_lag_s1", "W2T_lag_s2"],
     "EqFM distilled (teacher trained 6000 steps)": ["W2C_eqfm_distill_semigroup_wholejump", "W2S_eqfm_distill_s1", "W2S_eqfm_distill_s2"]}


def curve(run, K):
    pts = []
    for f in glob.glob(f"runs/{run}/frontier_s*.json"):
        if "tmp" in f: continue
        m = json.load(open(f))["steps"].get(K)
        if m: pts.append((m["entropy"], m["gen_ppl"]))
    return sorted(pts)


def at(pts, h):
    for (h0, p0), (h1, p1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            return p0 if h1 == h0 else p0 + (p1 - p0) * (h - h0) / (h1 - h0)
    return None


for K, targets in (("1", (3.3, 3.4, 3.5, 3.6, 3.7)), ("8", (3.5, 3.6, 3.7, 3.8, 3.9, 4.0))):
    print(f"\n| {K} evaluation(s) | seeds | unscaled noise | " + " | ".join(f"at entropy {h}" for h in targets) + " |\n|---|---|---|" + "---|" * len(targets))
    for g, runs in G.items():
        C = [c for c in (curve(r, K) for r in runs) if c]
        base = []
        for r in runs:
            f = f"runs/{r}/frontier_s1.0.json"
            if os.path.exists(f): base.append(json.load(open(f))["steps"][K])
        b = f"{st.mean(x['gen_ppl'] for x in base):.0f} ± {st.pstdev(x['gen_ppl'] for x in base):.0f} / {st.mean(x['entropy'] for x in base):.2f}" if base else ""
        cells = []
        for h in targets:
            v = [at(c, h) for c in C]
            if all(x is not None for x in v) and v:
                cells.append(f"{st.mean(v):.0f} ± {st.pstdev(v):.0f}")
            else:
                cells.append(f"not reached ({min(c[0][0] for c in C):.2f}–{max(c[-1][0] for c in C):.2f})" if C else "")
        print(f"| {g} | {len(C)} | {b} | " + " | ".join(cells) + " |")
