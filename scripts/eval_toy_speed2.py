"""Smooth against binary speed laws for the continuous adaptive-speed sampler, same base speeds, on a toy denoiser.

    python scripts/eval_toy_speed2.py CKPT out=file.json [num=8000] [K=16,64]

smooth: psi(g) = base + beta g^p          binary: psi(g) = base + 10^6 1[g > h]  (commit at once above the threshold h)
Prints valid sequences % / pair TV for every law and, per K and base, the best of each family."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from deqfm.clocks import build_clock
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample_speed_clock

path = sys.argv[1]
opt = dict(a.split("=", 1) for a in sys.argv[2:])
num = int(opt.get("num", 8000)); Ks = [int(v) for v in opt.get("K", "16,64").split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(path, map_location="cpu", weights_only=False); cfg = ckpt["cfg"]
data = build_dataset(cfg["data"])
model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"]).eval().to(dev)
model.load_state_dict(ckpt["ema"])
clock, cond = build_clock(cfg["clock"]), cfg["model"]["cond"]
rows = []
for K in Ks:
    for base in (0.5, 0.75, 0.9, 1.0):
        laws = [("binary", dict(threshold=h, beta=1e6)) for h in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
        laws += [("smooth", dict(beta=b, power=p)) for b in (256, 1024, 4096, 16384) for p in (8, 16, 32)]
        for fam, kw in laws:
            tok = torch.cat([sample_speed_clock(model, cond, clock, 2000, K, base=base, device=dev, **kw).cpu() for _ in range(num // 2000)]) if not torch.manual_seed(0) is None else None
            m = data.metrics(tok)
            rows.append({"K": K, "base": base, "family": fam, **{k: v for k, v in kw.items()}, **m})
        for fam in ("binary", "smooth"):
            b = max((r for r in rows if r["K"] == K and r["base"] == base and r["family"] == fam), key=lambda r: r["valid"])
            law = f"h={b['threshold']}" if fam == "binary" else f"beta={b['beta']:g}, p={b['power']:g}"
            print(f"K={K} base={base} best {fam}: {100 * b['valid']:.1f} / {b['pair_tv']:.3f} ({law})", flush=True)
json.dump({"ckpt": path, "step": ckpt["step"], "num": num, "rows": rows}, open(opt["out"], "w"), indent=1)
