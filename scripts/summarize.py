"""Tables of generative perplexity / entropy from runs/*/eval*.json.   python scripts/summarize.py [tasks.tsv ...]"""
import glob
import json
import os
import sys

NAMES = {}   # run name -> plain description, read from jobs/names.tsv
if os.path.exists("jobs/names.tsv"):
    for line in open("jobs/names.tsv"):
        if line.strip() and not line.startswith("#"):
            k, v = line.rstrip("\n").split("\t", 1)
            NAMES[k] = v


def cell(d):
    return f"{d['gen_ppl']:.1f} / {d['entropy']:.2f}" if d else ""


def table(title, rows, cols):
    print(f"\n### {title}\n\n(generative perplexity under GPT-2 Large, lower is better / entropy, real data 4.31)\n")
    print("| Model | Training steps | " + " | ".join(f"{c} steps" if c != "data" else "value" for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name, path in rows:
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        print(f"| {NAMES.get(name, name)} | {r.get('train_step') or ''} | " + " | ".join(cell(r["steps"].get(str(c))) for c in cols) + " |")


def main():
    B = "runs/baselines"
    table("References", [("Real LM1B test sequences", f"{B}/data.json")], ["data"])
    table("Released FMLM (flow map)", [
        ("FMLM released, deterministic sampler", f"{B}/fmlm_released.json"),
        ("FMLM released, stochastic sampler (churn 0.8, as in the paper)", f"{B}/fmlm_released_churn0.8.json"),
        ("FMLM released, time read from the state instead of given", f"{B}/fmlm_released_statetime.json")], [1, 2, 4, 8, 16, 32])
    table("Released FLM (denoiser, Euler sampler)", [
        ("FLM released", f"{B}/flm_released.json"),
        ("FLM released, time read from the state instead of given", f"{B}/flm_released_statetime.json")], [8, 16, 32, 64, 128, 256, 1024])
    files = sys.argv[1:] or sorted(glob.glob("jobs/tasks_*.tsv"))
    for tasks in files:
        fields, maps = [], []
        for line in open(tasks):
            if not line.strip() or line.startswith("#"):
                continue
            name, _, script, _ = line.rstrip("\n").split("\t")
            for path in sorted(glob.glob(f"runs/{name}/eval*.json")):
                tag = os.path.basename(path)[4:-5].strip("_")
                (fields if "field" in script else maps).append((name + (f" [{tag}]" if tag else ""), path))
        if fields:
            table(f"{tasks}: denoisers (Euler sampler)", fields, [8, 32, 128, 1024])
        if maps:
            table(f"{tasks}: flow maps", maps, [1, 2, 4, 8])


if __name__ == "__main__":
    main()
