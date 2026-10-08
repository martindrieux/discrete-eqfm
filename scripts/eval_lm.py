"""Generative perplexity and entropy of samples from a checkpoint, for several numbers of sampling steps.

    python scripts/eval_lm.py CKPT out=results.json [eval.steps=[1,2,4,8]] [eval.num=1024] [eval.grid=reparam]
                              [churn=0.0] [noise_scale=1.0] [sampler=refine threshold=0.9] [cond=...] [flow_map=true|false] [seed=1]

CKPT is a checkpoint written by this repository, or one of
    released:flm:<dir>    released:fmlm:<dir>     (FLM-B-LM1B / FMLM-B-LM1B from the FLM authors, model.safetensors)
    data:<blocks file>                            (real sequences: the reference values)
`cond` overrides how the network is told the time (deqfm/nets.py); e.g. cond=state_time runs a time-conditioned
checkpoint with the time read off the state.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from deqfm.clocks import build_clock
from deqfm.config import load_config
from deqfm.evaluation import text_metrics
from deqfm.sampling import sample_grid, sample_refine, time_grid
from deqfm.trainer import load_model


def main():
    src = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    cfg = load_config(None, [a for a in sys.argv[2:] if a.startswith("eval.")])
    E = cfg.eval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(opts.get("seed", 1)))
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoTokenizer
    result = {"source": src, "eval": dict(E), "opts": opts, "steps": {}}

    if src.startswith("data:"):
        path = src[5:]
        data = np.load(path, mmap_mode="r")
        idx = np.sort(np.random.default_rng(0).choice(len(data), E.num, replace=False))
        tokens = torch.from_numpy(data[idx].astype(np.int64))
        tok = AutoTokenizer.from_pretrained(json.load(open(path + ".json"))["tokenizer"])
        result["steps"]["data"], texts = text_metrics(tokens, tok, E.judge, device)
        result["samples"] = texts[:8]
    else:
        model, ckpt = load_model(src, device, use_ema=opts.get("ema", "true") == "true")
        c = ckpt["cfg"]
        cond, flow_map, clock, step = c["model"]["cond"], ckpt["kind"] == "flow_map", build_clock(c["clock"]), ckpt["step"]
        tok_name = "bert-base-uncased" if src.startswith("released:") else json.load(open(c["data"]["path"] + ".json"))["tokenizer"]
        if not flow_map and not any(a.startswith("eval.steps") for a in sys.argv[2:]):
            E["steps"] = [8, 32, 128, 1024]   # a field is integrated with many small steps
        cond = opts.get("cond", cond)
        flow_map = opts.get("flow_map", str(flow_map)).lower() == "true"
        churn = float(opts.get("churn", 0.0))
        tok = AutoTokenizer.from_pretrained(tok_name)
        result.update(cond=cond, flow_map=flow_map, churn=churn, train_step=step)
        for steps in E.steps:
            tic = time.time()
            grid = time_grid(steps, E.grid, model.vocab_size)
            if opts.get("sampler") == "refine":      # DBTM sampler: `steps` rounds of commit and renoise
                tokens = torch.cat([sample_refine(model, cond, clock, min(E.batch, E.num - i), steps,
                                                  float(opts.get("threshold", 0.9)), device).cpu()
                                    for i in range(0, E.num, E.batch)])
            else:
                tokens = torch.cat([sample_grid(model, cond, clock, min(E.batch, E.num - i), grid, flow_map, churn, device,
                                                noise_scale=float(opts.get("noise_scale", 1.0))).cpu()
                                    for i in range(0, E.num, E.batch)])
            result["steps"][str(steps)], texts = text_metrics(tokens, tok, E.judge, device)
            result["steps"][str(steps)]["seconds"] = time.time() - tic
            result.setdefault("samples", {})[str(steps)] = texts[:4]
            print(f"{steps:>5d} steps | gen ppl {result['steps'][str(steps)]['gen_ppl']:.2f} | "
                  f"entropy {result['steps'][str(steps)]['entropy']:.3f} | {time.time() - tic:.0f} s", flush=True)
            if "out" in opts:
                os.makedirs(os.path.dirname(os.path.abspath(opts["out"])), exist_ok=True)
                with open(opts["out"], "w") as f:
                    json.dump(result, f, indent=1)
    print(json.dumps(result["steps"], indent=1))
    if "out" in opts:
        os.makedirs(os.path.dirname(os.path.abspath(opts["out"])), exist_ok=True)
        with open(opts["out"], "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
