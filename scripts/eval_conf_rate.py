"""Samples a denoiser with the confidence-rate sampler (deqfm.sampling.sample_conf_rate) and with continuous adaptive
speed, for several numbers of network evaluations K, and writes the metrics.

    python scripts/eval_conf_rate.py CKPT out=results.json [K=8,16,32,256] [beta=0,4] [lam=4] [s_max=4]
                                     [speed_beta=0,1,4,16] [num=...] [ema=true]

Samplers, K evaluations each:
    conf   x^l <- (1 - s^l) x^l + s^l D(x)^l, s^l = 1 - exp(-h a(x^l)), a = 1 + beta exp(-r(x^l) / lambda), h = s_max / K
           (beta = 0: the constant rate; one row per beta)
    speed  continuous adaptive speed in the decoding-error clock, psi(g) = 1 + beta g with g = max_v D(x)^l_v
           (sample_speed_clock, base 1, power 1; beta = 0: tied clocks, the plain Euler sampler on the decoding-clock grid)
Markov toy (data.name = markov): exact metrics of the dataset (valid, pair TV, ...). Text (data.name = blocks):
GPT-2 Large perplexity and token-histogram entropy, as scripts/eval_lm.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deqfm.clocks import build_clock
from deqfm.data import build_dataset
from deqfm.sampling import sample_conf_rate, sample_speed_clock
from deqfm.trainer import load_model


def main():
    path = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    floats = lambda k, d: [float(v) for v in opt.get(k, d).split(",") if v != ""]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, ckpt = load_model(path, dev, use_ema=opt.get("ema", "true") == "true")
    c = ckpt["cfg"]
    cond, clock = c["model"]["cond"], build_clock(c["clock"])
    toy = c["data"]["name"] == "markov"
    data = build_dataset(c["data"])
    num = int(opt.get("num", 4000 if toy else 1024))
    batch = num if toy else 128
    lam = float(opt.get("lam", c["loss"].get("conf_lambda", 1.0)))
    s_max = float(opt.get("s_max", 4.0))
    result = {"ckpt": path, "train_step": ckpt["step"], "lam": lam, "s_max": s_max, "num": num, "rows": []}
    if toy:
        result["real"] = data.metrics(data.draw(num))
    else:
        from transformers import AutoTokenizer
        from deqfm.evaluation import text_metrics
        tok = AutoTokenizer.from_pretrained(data.meta["tokenizer"])

    def run(sampler):
        torch.manual_seed(int(opt.get("seed", 0)))
        return torch.cat([sampler(min(batch, num - i)).cpu() for i in range(0, num, batch)])

    def add(name, K, beta, tokens):
        m = data.metrics(tokens) if toy else text_metrics(tokens, tok, c["eval"]["judge"], dev)[0]
        result["rows"].append({"sampler": name, "K": K, "beta": beta, **m})
        keys = ("valid", "pair_tv", "distinct") if toy else ("gen_ppl", "entropy")
        print(f"{name:6s} K={K:4d} beta={beta:g}: " + " ".join(f"{k} {m[k]:.4f}" for k in keys), flush=True)
        if "out" in opt:
            with open(opt["out"], "w") as f:
                json.dump(result, f, indent=1)

    for K in [int(v) for v in opt.get("K", "8,16,32,256").split(",")]:
        for beta in floats("beta", "0,4"):
            add("conf", K, beta, run(lambda n: sample_conf_rate(model, cond, clock, n, K, s_max, beta, lam, dev)))
        for beta in floats("speed_beta", "0,1,4,16"):
            add("speed", K, beta, run(lambda n: sample_speed_clock(model, cond, clock, n, K, beta, 1.0, device=dev)))


if __name__ == "__main__":
    main()
