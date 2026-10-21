"""Record toy runs (Markov chain, Section 7 of the notes) in the nb tracker, once each.

    python scripts/nb_record_toy.py runs/TM4_long_plain runs/TM5_plain_s1 ...

Metrics, for K network evaluations (decoding-clock grid): valid_K = share of samples whose transitions are all allowed,
tv_K = total variation distance between sampled and true pair frequencies; suffix _commit = jumps (or Euler steps) with
confident positions committed (best threshold), _refine = map to the end with commits and kept noise (flow maps).
Runs that differ only by the seed get the same title, formula and settings, so the site shows them as one row.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nb

import deqfm.losses, deqfm.reparam, deqfm.sampling  # noqa: E401,F401

SUB = "toy-commit-states"
QUESTION = ("On a sparse Markov chain (exact metrics), does training on states with clean (committed) positions, random "
            "(Section 7.1) or chosen by the model's confidence (Section 7.2), improve few-step samples?")


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def describe(cfg, field):
    L = "loss."
    kind = "Denoiser" if field else "Flow map"
    if cfg[L + "adaptive"] == "speed_clock":
        what = "continuous adaptive speed, sharp law" + (", mixed with random clean positions" if cfg[L + "partial_context"] else "")
        f = r"progress $u^l$ in the decoding-error clock, $\dot u^l = b + \beta (g^l)^p$, $b$ = \cfg{loss.adaptive_base}, $\beta$ = \cfg{loss.adaptive_beta}, $p$ = \cfg{loss.adaptive_power}, $g^l = \max_i \mathrm{sg}[\hat D(x)]^{l,i}$, model in the loop, share of samples \cfg{loss.adaptive_frac}"
    elif cfg[L + "adaptive"] == "speed":
        what = "one time per position"
        f = r"$\dot t^l = 1 + \beta g^l$, $\beta$ = \cfg{loss.adaptive_beta}, $g^l = \max_i p^{l,i}$, share of samples \cfg{loss.adaptive_frac}"
    elif cfg[L + "adaptive"] in ("binary", "rollout"):
        what = "model-chosen clean positions" + (", step by step" if cfg[L + "adaptive"] == "rollout" else "")
        f = r"$C = \{l : \max_i p^{l,i} > h\}$, $h$ = \cfg{loss.adaptive_h}, $x^l = x_1^l$ on $C$, share of samples \cfg{loss.adaptive_frac}"
    elif cfg.get(L + "random_speed", 0) > 0:
        what = "random speed per position"
        f = r"$t^l = \min(1, t(1 + \beta g^l))$, $g^l \sim \mathcal{U}[0,1]$, $\beta \sim \mathcal{U}[0, \beta_{\max}]$, $\beta_{\max}$ = \cfg{loss.random_speed}, share of samples \cfg{loss.context_frac}"
    elif cfg[L + "partial_context"]:
        what = "random clean positions"
        f = r"$l \in C$ with probability $f \sim \mathcal{U}[0, 1]$, $x^l = x_1^l$ on $C$, share of samples \cfg{loss.context_frac}"
    else:
        what = "no clean positions"
        f = r"$x = I_t$ (Algorithm 1 of the notes)"
    return f"{kind}, {what}", f + r"; \cfg{train.steps} steps"


def record(d):
    name = os.path.basename(d.rstrip("/"))
    ev = f"{d}/toy_eval_reparam.json" if os.path.exists(f"{d}/toy_eval_reparam.json") else f"{d}/toy_eval.json"
    sweep = f"{d}/speed_sweep.json"       # continuous adaptive-speed sampler: valid_K_speed = best speed law
    if not os.path.exists(ev) and os.path.exists(sweep):
        ev = sweep
    if not os.path.exists(ev):
        return print(f"{name}: no evaluation, skipped")
    if os.path.exists(f"{d}/NB_RECORDED"):
        return print(f"{name}: already recorded")
    cfg = flat(json.load(open(f"{d}/config.json")))
    for k in ("out_dir", "ckpt_dir", "train.resume", "train.max_minutes"):
        cfg.pop(k, None)
    r = json.load(open(ev))
    for x in r["rows"]:
        x["sampler"] = {"commit": "adaptive"}.get(x["sampler"], x["sampler"])
    field = not any(x["sampler"].startswith("refine") for x in r["rows"])
    title, formula = describe(cfg, field)
    run = nb.start(subproject=SUB, title=title, formula=formula, algo="denoiser" if field else "flow-map", config=cfg, question=QUESTION)
    last = 0
    for line in open(f"{d}/log.jsonl"):
        x = json.loads(line)
        if x["step"] > last:
            last = x["step"]
            run.log(step=x["step"], loss=x["loss"])
    metrics = {}
    for K in sorted({x["K"] for x in r["rows"]}):
        for sampler, suffix in (("plain", ""), ("adaptive", "_commit"), ("refine-keep", "_refine"), ("speed", "_speed")):
            rows = [x for x in r["rows"] if x["K"] == K and x["sampler"] == sampler]
            if rows:
                b = max(rows, key=lambda x: x["valid"] - x["pair_tv"])
                metrics[f"valid_{K}{suffix}"], metrics[f"tv_{K}{suffix}"] = b["valid"], b["pair_tv"]
    run.log(step=last, **metrics)
    run.finish()
    open(f"{d}/NB_RECORDED", "w").close()
    print(f"{name}: recorded as '{title}'")


for d in sys.argv[1:]:
    try:
        record(d)
    except Exception as e:      # noqa: BLE001
        print(f"{d}: not recorded ({type(e).__name__}: {e})")
