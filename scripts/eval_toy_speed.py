"""Continuous adaptive-speed sampler on toy denoisers: sweep of the speed law psi(g) = 1 + beta g^power.

    python scripts/eval_toy_speed.py CKPT out=file.json [num=4000] [K=8,16,64] [beta=4,16,64,256] [power=1,2,4,8,16]

Prints, per number of evaluations K: the plain Euler sampler, the sampler with commits (best threshold), and every
(beta, power). Cells: valid sequences % / pair TV."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from deqfm.clocks import build_clock
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample_adaptive, sample_grid, sample_speed_clock, time_grid


def main():
    path = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    num = int(opt.get("num", 4000))
    Ks = [int(v) for v in opt.get("K", "8,16,64").split(",")]
    betas = [float(v) for v in opt.get("beta", "4,16,64,256").split(",")]
    powers = [float(v) for v in opt.get("power", "1,2,4,8,16").split(",")]
    extra = {k: float(v) for k, v in opt.items() if k in ("floor", "base")}
    snap = opt.get("snap", "true") == "true"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    data = build_dataset(cfg["data"])
    model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"]).eval().to(dev)
    model.load_state_dict(ckpt["ema"])
    clock, cond = build_clock(cfg["clock"]), cfg["model"]["cond"]
    rows = []

    def add(name, K, tok, **kw):
        m = data.metrics(tok)
        rows.append({"sampler": name, "K": K, **kw, **m})
        return f"{100 * m['valid']:.1f} / {m['pair_tv']:.3f}"

    print(f"checkpoint {path} step {ckpt['step']}")
    for K in Ks:
        grid = time_grid(K, "reparam", ckpt["vocab_size"])
        torch.manual_seed(0)
        plain = add("plain", K, sample_grid(model, cond, clock, num, grid, flow_map=False, device=dev))
        best = ""
        for h in (0.5, 0.7, 0.9):
            torch.manual_seed(0)
            c = add("commit", K, sample_adaptive(model, cond, clock, num, grid, h=h, flow_map=False, device=dev)[0], h=h)
            best = max(best, c, key=lambda s: float(s.split(" /")[0]) if s else -1)
        print(f"\nK = {K}: plain {plain} | commits (best threshold) {best}\n| beta \\ power | " + " | ".join(f"{p:g}" for p in powers) + " |\n|---|" + "---|" * len(powers))
        for b in betas:
            cells = []
            for p in powers:
                torch.manual_seed(0)
                cells.append(add("speed", K, sample_speed_clock(model, cond, clock, num, K, beta=b, power=p, device=dev, snap=snap, **extra), beta=b, power=p))
            print(f"| {b:g} | " + " | ".join(cells) + " |", flush=True)
    if "out" in opt:
        json.dump({"ckpt": path, "step": ckpt["step"], "rows": rows}, open(opt["out"], "w"), indent=1)


if __name__ == "__main__":
    main()
