"""Perplexity / entropy of WikiText-2 maps for each scale of the initial noise.   python scripts/frontier_table.py [steps]"""
import glob, json, os, sys
K = sys.argv[1] if len(sys.argv) > 1 else "1"
NAMES = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open("jobs/names.tsv") if l.strip() and not l.startswith("#")}
S = ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.35", "1.5"]
print(f"| Model ({K} evaluation(s)) | " + " | ".join(f"noise x{s}" for s in S) + " |\n|---|" + "---|" * len(S))
for d in sorted({os.path.dirname(p) for p in glob.glob("runs/W2*/frontier_s*.json")}):
    cells = []
    for s in S:
        f = f"{d}/frontier_s{s}.json"
        m = json.load(open(f))["steps"].get(K) if os.path.exists(f) else None
        cells.append(f"{m['gen_ppl']:.0f} / {m['entropy']:.2f}" if m else "")
    print(f"| {NAMES.get(os.path.basename(d), os.path.basename(d))} | " + " | ".join(cells) + " |")
