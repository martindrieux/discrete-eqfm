"""Training loop shared by both objectives. Multi-GPU and mixed precision come from `accelerate`."""
import copy
import json
import math
import os
import signal
import sys
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .config import to_dict


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        decay = min(self.decay, (1 + step) / (10 + step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - decay)


def load_checkpoint(path, model, use_ema=True, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    return ckpt


def load_released(path, device="cpu"):
    """A checkpoint released by the FLM authors (directory with model.safetensors): FLM-B-LM1B or FMLM-B-LM1B."""
    from safetensors.torch import load_file
    from .dit import DiT
    sd = load_file(os.path.join(path, "model.safetensors"))
    sd = {k.replace("backbone.", "").replace("._orig_mod", ""): v for k, v in sd.items() if "rotary_emb" not in k}
    two = any(k.startswith("sigma_map_prime") for k in sd)
    model = DiT(sd["vocab_embed.embedding"].shape[0], 128, second_input=two)
    model.load_state_dict(sd)
    return model.to(device).eval().requires_grad_(False), two


def load_model(path, device="cpu", use_ema=True):
    """Rebuild the network of a checkpoint written by `train`. Returns (model in eval mode, checkpoint dict).

    `released:<dir>` loads a checkpoint of the FLM authors; it is time-conditioned (model.cond = time)."""
    from .config import load_config
    from .model import build_model
    if path.startswith("released:"):
        model, two = load_released(path.split(":")[-1], device)
        cfg = load_config(None, ["model.arch=dit", "model.dim=768", "model.depth=12", "model.heads=12", "model.cond=time"])
        return model, {"cfg": cfg, "kind": "flow_map" if two else "field", "step": None,
                       "vocab_size": model.vocab_size, "seq_len": model.seq_len}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config(None)
    for key, value in ckpt["cfg"].items():   # defaults for keys added after the checkpoint was written
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    ckpt["cfg"] = cfg
    model = build_model(cfg["model"], ckpt["vocab_size"], ckpt["seq_len"], ckpt["kind"])
    model.load_state_dict(ckpt["ema" if use_ema and "ema" in ckpt else "model"])
    return model.to(device).eval().requires_grad_(False), ckpt


def _atomic_save(state, path):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(cfg, model, dataset, loss_fn, kind, teacher=None):
    """loss_fn(model, raw, tokens) -> (loss, logs). `kind` ('field' or 'flow_map') is stored in the checkpoint."""
    T = cfg.train
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(mixed_precision=T.precision, gradient_accumulation_steps=T.grad_accum)
    is_main = accelerator.is_main_process
    ckpt_dir = cfg.ckpt_dir or cfg.out_dir
    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(to_dict(cfg), f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, betas=tuple(T.betas), weight_decay=T.weight_decay)
    raw = model
    ema = EMA(raw, T.ema)

    step, last = 0, os.path.join(ckpt_dir, "last.pt")
    if T.resume and os.path.exists(last):
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        raw.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        del ckpt
        accelerator.print(f"resumed from {last} at step {step}")
    set_seed(cfg.seed + step, device_specific=True)   # a resumed run draws new batches, not the first ones again
    loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True, drop_last=True,
                        num_workers=T.num_workers, pin_memory=True, persistent_workers=T.num_workers > 0)

    # Preemption (SIGTERM) and the time budget both end the run with a checkpoint and exit code 3.
    # The flag is shared between processes through the accelerator, so that all of them stop at the same step.
    stop = {"signal": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("signal", True))

    model, opt, loader = accelerator.prepare(model, opt, loader)
    ema.model.to(accelerator.device)
    loss_fn.ema = ema.model      # used when loss.target_net = ema
    if teacher is not None:
        teacher.to(accelerator.device).eval().requires_grad_(False)

    run = None
    if is_main and cfg.wandb.project:
        import wandb
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=to_dict(cfg))

    def save(name, full=True):
        state = {"ema": ema.model.state_dict(), "step": step, "cfg": to_dict(cfg), "kind": kind,
                 "vocab_size": dataset.vocab_size, "seq_len": dataset.seq_len}
        if full:
            state.update(model=raw.state_dict(), opt=opt.state_dict())
        _atomic_save(state, os.path.join(ckpt_dir, name))

    n_params = sum(p.numel() for p in raw.parameters())
    accelerator.print(f"{kind}: {n_params / 1e6:.2f}M parameters, vocab {dataset.vocab_size}, "
                      f"length {dataset.seq_len}, {accelerator.num_processes} process(es)")

    model.train()
    first_step, rate_tic = step, time.time()
    tic, logs_sum, logs_n = time.time(), {}, 0
    data = iter(loader)
    while step < T.steps:
        try:
            tokens = next(data)
        except StopIteration:
            data = iter(loader)
            tokens = next(data)
        lr = T.lr * min(1.0, (step + 1) / max(1, T.warmup))
        if T.lr_decay == "cosine":
            lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / T.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss, logs = loss_fn(model, raw, tokens)
            accelerator.backward(loss)
            if accelerator.sync_gradients and T.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), T.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for k, v in logs.items():
            logs_sum[k] = logs_sum.get(k, 0.0) + v
        logs_n += 1
        if not accelerator.sync_gradients:
            continue
        step += 1
        ema.update(raw, step)
        if stop["signal"] or (is_main and T.max_minutes and time.time() - started > 60 * T.max_minutes):
            accelerator.set_trigger()
        if is_main and T.min_rate and step - first_step in (100, 400):      # some nodes synchronise GPUs very slowly
            if step - first_step == 100:
                rate_tic = time.time()
            elif 300 / (time.time() - rate_tic) < T.min_rate:
                print(f"slow node: {300 / (time.time() - rate_tic):.2f} it/s < {T.min_rate}", flush=True)
                open(os.path.join(cfg.out_dir, "SLOW"), "w").close()
                accelerator.set_trigger()
        if accelerator.check_trigger():
            if is_main and step < T.steps:
                save("last.pt")
                open(os.path.join(cfg.out_dir, "INTERRUPTED"), "w").close()   # a multi-GPU launcher hides the exit code
                print(f"interrupted at step {step}: checkpoint written, exiting with code 3", flush=True)
            if step < T.steps:
                accelerator.wait_for_everyone()
                sys.exit(3)
        if not is_main:
            continue
        if step % T.log_every == 0 or step == T.steps:
            avg = {k: v / logs_n for k, v in logs_sum.items()}
            rate = logs_n / (time.time() - tic)
            print(f"step {step:>8d} | " + " | ".join(f"{k} {v:.4f}" for k, v in avg.items())
                  + f" | lr {lr:.2e} | {rate:.2f} it/s", flush=True)
            with open(os.path.join(cfg.out_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "lr": lr, **avg}) + "\n")
            if run is not None:
                run.log(avg, step=step)
            if os.environ.get("DEQ_REPORT_MEM") and torch.cuda.is_available():
                print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            tic, logs_sum, logs_n = time.time(), {}, 0
        if step % T.save_every == 0 or step == T.steps:
            save("last.pt")
        if T.keep_every and step % T.keep_every == 0 and step < T.steps:
            save(f"step_{step:07d}.pt", full=False)
    accelerator.wait_for_everyone()
    return raw, ema.model
