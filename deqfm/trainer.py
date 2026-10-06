"""Training loop shared by both objectives. Multi-GPU and mixed precision come from `accelerate`."""
import copy
import json
import os
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


def train(cfg, model, dataset, loss_fn, kind, teacher=None):
    """loss_fn(model, raw, tokens) -> (loss, logs). `kind` ('field' or 'flow_map') is stored in the checkpoint."""
    T = cfg.train
    accelerator = Accelerator(mixed_precision=T.precision, gradient_accumulation_steps=T.grad_accum)
    set_seed(cfg.seed, device_specific=True)
    is_main = accelerator.is_main_process
    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(to_dict(cfg), f, indent=2)

    loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=True, drop_last=True,
                        num_workers=T.num_workers, pin_memory=True, persistent_workers=T.num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, betas=tuple(T.betas), weight_decay=T.weight_decay)
    raw = model
    ema = EMA(raw, T.ema)

    step, last = 0, os.path.join(cfg.out_dir, "last.pt")
    if T.resume and os.path.exists(last):
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        raw.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        accelerator.print(f"resumed from {last} at step {step}")

    model, opt, loader = accelerator.prepare(model, opt, loader)
    ema.model.to(accelerator.device)
    if teacher is not None:
        teacher.to(accelerator.device).eval().requires_grad_(False)

    run = None
    if is_main and cfg.wandb.project:
        import wandb
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=to_dict(cfg))

    def save(name):
        state = {"model": raw.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                 "step": step, "cfg": to_dict(cfg), "kind": kind,
                 "vocab_size": dataset.vocab_size, "seq_len": dataset.seq_len}
        torch.save(state, os.path.join(cfg.out_dir, name))

    n_params = sum(p.numel() for p in raw.parameters())
    accelerator.print(f"{kind}: {n_params / 1e6:.2f}M parameters, vocab {dataset.vocab_size}, "
                      f"length {dataset.seq_len}, {accelerator.num_processes} process(es)")

    model.train()
    tic, logs_sum, logs_n = time.time(), {}, 0
    data = iter(loader)
    while step < T.steps:
        try:
            tokens = next(data)
        except StopIteration:
            data = iter(loader)
            tokens = next(data)
        lr = T.lr * min(1.0, (step + 1) / max(1, T.warmup))
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
            tic, logs_sum, logs_n = time.time(), {}, 0
        if step % T.save_every == 0 or step == T.steps:
            save("last.pt")
            if step % T.save_every == 0:
                save(f"step_{step:07d}.pt")
    accelerator.wait_for_everyone()
    return raw, ema.model
