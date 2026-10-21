"""WikiText-2 tables.   python scripts/summarize_w2.py jobs/tasks_w2.tsv [jobs/tasks_w2b.tsv ...]
One row per run: best held-out loss and its step, then perplexity / entropy of the samples of that checkpoint for each
sampler and number of network evaluations. Real test text: perplexity 41.7, entropy 4.16."""
import json
import os
import sys

NAMES = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open("jobs/names.tsv") if l.strip() and not l.startswith("#")}
SAMPLERS = [("eval", "jumps"), ("eval_churn0.8", "jumps, churn 0.8"), ("eval_refine", "commit, fresh noise"),
            ("eval_refine_rep1", "commit, fresh noise, repetition penalty"), ("eval_keepnoise", "commit, same noise"),
            ("eval_adaptive_h0.5", "steps with commits, threshold 0.5"), ("eval_adaptive_h0.7", "steps with commits, threshold 0.7"),
            ("eval_adaptive_h0.9", "steps with commits, threshold 0.9")]


def cell(m):
    return f"{m['gen_ppl']:.0f} / {m['entropy']:.2f}" if m else ""


for tasks in sys.argv[1:]:
    print(f"\n### {tasks}\n\n| Model | Held-out loss (step) | Sampler | " + " | ".join(str(k) for k in (1, 2, 4, 8, 16, 32, 128, 1024)) + " |")
    print("|---|---|---|" + "---|" * 8)
    for line in open(tasks):
        if not line.strip() or line.startswith("#"):
            continue
        name = line.split("\t")[0]
        d = f"runs/{name}"
        val = ""
        if os.path.exists(f"{d}/val.jsonl"):
            v = [json.loads(l) for l in open(f"{d}/val.jsonl")]
            b = min(v, key=lambda r: r["val"])
            val = f"{b['val']:.2f} ({b['step']})"
        first = True
        for tag, label in SAMPLERS:
            if not os.path.exists(f"{d}/{tag}.json"):
                continue
            s = json.load(open(f"{d}/{tag}.json"))["steps"]
            label = "Euler steps" if tag == "eval" and "32" in s else label
            print(f"| {NAMES.get(name, name) if first else ''} | {val if first else ''} | {label} | " +
                  " | ".join(cell(s.get(str(k))) for k in (1, 2, 4, 8, 16, 32, 128, 1024)) + " |")
            first = False
