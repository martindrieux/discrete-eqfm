"""Record finished runs in the nb experiment tracker: one experiment per run, written when the run is complete.

    python scripts/nb_record.py NAME [NAME ...]        (run names under runs/; `--all` for every finished run)
    python scripts/nb_record.py --stopped NAME [...]   (a run stopped before its last step, with its kept checkpoints)

A run is trained in several jobs (time limits, preemption) and nb.start cannot resume an experiment, so a run is
recorded once at the end: the training curves come from runs/NAME/log.jsonl and the sample metrics from
runs/NAME/eval*.json. Titles and subprojects are read from jobs/names.tsv (name, subproject, title).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nb

import deqfm.losses, deqfm.reparam, deqfm.sampling  # noqa: E401,F401  the @nb.core functions must be imported before nb.start

COLLAPSE_ENTROPY = 3.9      # real LM1B text: 4.31

QUESTIONS = {
    "denoiser-training": "Does a denoiser without a time input train as well as a time-conditioned one, and which law of $t$ is best?",
    "flow-map-from-data": "Which off-diagonal loss, clock and sampling of jumps train a flow map from data alone?",
    "flow-map-distillation": "Which off-diagonal loss and which network inputs distil a denoiser into the best one-step flow map?",
    "released-flm-distillation": "Starting from the FLM checkpoint of its authors, does a time-free flow map match FMLM?",
    "endpoint-only": "Does learning only the whole jump $\\kappa = 1$ (as in Discrete Beckmann Transport Models) work in this framework?",
}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def formula(kind):
    if kind == "denoiser":
        return (r"$\hat D$ with inputs \cfg{model.cond}, $t$ by \cfg{loss.diag_time}, "
                r"\cfg{model.depth} blocks of width \cfg{model.dim}, \cfg{train.steps} steps")
    return (r"$\bar\delta$ by \cfg{loss.off_diagonal} (whole jump: \cfg{loss.whole_jump_target}), inputs \cfg{model.cond}, "
            r"jumps by \cfg{loss.jump_sampling}, boundary probability \cfg{loss.boundary_prob}, clock \cfg{clock}, "
            r"teacher \cfg{loss.teacher}, \cfg{train.steps} steps")


def record(name, names, stopped=False):
    d = f"runs/{name}"
    if not os.path.exists(f"{d}/DONE") and not stopped:
        return print(f"{name}: not finished, skipped (use --stopped NAME for a run that was stopped for good)")
    if os.path.exists(f"{d}/NB_RECORDED"):
        return print(f"{name}: already recorded")
    if name not in names:
        return print(f"{name}: add it to jobs/names.tsv first")
    sub, title = names[name]
    cfg = flat(json.load(open(f"{d}/config.json")))
    evals = {os.path.basename(p)[4:-5].strip("_"): json.load(open(p)) for p in sorted(glob.glob(f"{d}/eval*.json"))
             if ".tmp" not in p}
    kind = "flow-map" if any(e.get("flow_map") for e in evals.values()) or cfg.get("loss.off_diagonal") and "field" not in name else "denoiser"
    for k in ("loss.teacher", "train.init", "data.path", "out_dir", "ckpt_dir"):      # keep paths short
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].replace("/data/user_data/mdrieux/deqfm/", "")
    # Collapse: samples that repeat one token have entropy near 0 and a perplexity near 1, which the results table
    # would print as the best value. Such a perplexity is void: it is not logged, and the title says so.
    collapsed = sorted({int(k) for e in evals.values() for k, m in e["steps"].items()
                        if k.isdigit() and m["entropy"] < COLLAPSE_ENTROPY and (e.get("train_step") or 0) == max(
                            (x.get("train_step") or 0) for x in evals.values())})
    if collapsed:
        title = (title + ": collapsed at " + ", ".join(str(k) for k in collapsed) + " step" + ("s" if collapsed != [1] else ""))[:100]
    titles = json.load(open("jobs/titles.json"))      # name -> [title, formula]
    run = nb.start(subproject=sub, title=title, formula=titles.get(name, [title, formula(kind)])[1], algo=kind, config=cfg,
                   question=QUESTIONS[sub])
    last = 0
    for line in open(f"{d}/log.jsonl"):
        r = json.loads(line)
        if r["step"] <= last:      # a resumed job repeats a few steps
            continue
        last = r["step"]
        run.log(step=r["step"], **{k: v for k, v in r.items() if k in ("loss", "diag", "off")})
    for tag, e in sorted(evals.items(), key=lambda kv: kv[1].get("train_step") or last):   # kept checkpoints, then the end
        suffix = ("_churn" if "churn" in tag else "_refine" if "refine" in tag           # refine: DBTM sampler
                  else "_keepnoise" if "keepnoise" in tag else "")                        # same, noise kept between rounds
        metrics = {}
        for k, m in e["steps"].items():
            if suffix and k == "1":     # one step has no churn: same number as the deterministic sampler
                continue
            if m["entropy"] >= COLLAPSE_ENTROPY:
                metrics[f"gen_ppl_{k}{suffix}"] = m["gen_ppl"]
            metrics[f"entropy_{k}{suffix}"] = m["entropy"]
        run.log(step=e.get("train_step") or last, **metrics)
    run.finish()
    open(f"{d}/NB_RECORDED", "w").close()
    print(f"{name}: recorded in {sub}")


def main():
    names = {}
    for line in open("jobs/names.tsv"):
        if line.strip() and not line.startswith("#"):
            n, sub, title = line.rstrip("\n").split("\t")
            names[n] = (sub, title)
    args = sys.argv[1:]
    stopped = args[:1] == ["--stopped"]      # runs stopped before their last step: recorded with their kept checkpoints
    args = args[1:] if stopped else args
    if args == ["--all"]:
        args = [n for n in names if os.path.exists(f"runs/{n}/DONE")]
    for n in args:
        try:
            record(n, names, stopped)
        except Exception as e:      # noqa: BLE001  the tracker must never break the job pool
            print(f"{n}: not recorded ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
