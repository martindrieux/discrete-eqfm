"""Transformer denoiser on one-hot states.

Input : x in R^{B x L x V} (a noisy one-hot matrix) and a duration sigma in [0, 1]^B.
Output: logits in R^{B x L x V}; softmax over the last axis gives delta_sigma(x), a point of the product simplex.

sigma = 0 is the denoiser D (the autonomous field is D(x) - x). The model has no time input.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(sigma, dim, scale):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=sigma.dtype) / half)
    args = scale * sigma[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, manual=False):
        B, L, C = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        if manual:  # plain matmul attention: supports forward-mode differentiation
            a = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            out = a.softmax(dim=-1) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class Block(nn.Module):
    """Pre-norm transformer block; the duration embedding sets the scale and shift of both norms."""

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.mod = nn.Linear(dim, 4 * dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, manual=False):
        s1, b1, s2, b2 = self.mod(cond)[:, None].chunk(4, dim=-1)
        x = x + self.drop(self.attn(self.norm1(x) * (1 + s1) + b1, manual))
        x = x + self.drop(self.mlp(self.norm2(x) * (1 + s2) + b2))
        return x


class Denoiser(nn.Module):
    def __init__(self, vocab_size, seq_len, dim=256, depth=4, heads=4, mlp_ratio=4, dropout=0.0, sigma_scale=1.0):
        super().__init__()
        self.vocab_size, self.seq_len, self.dim, self.sigma_scale = vocab_size, seq_len, dim, sigma_scale
        self.embed = nn.Linear(vocab_size, dim, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.cond = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.pos, std=0.02)
        for layer in (self.mod, self.head):  # start from the uniform prediction
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, sigma, manual_attn=False):
        """x: [B, L, V] float, sigma: [B] float. Returns logits [B, L, V]."""
        h = self.embed(x) + self.pos[:, :x.shape[1]]
        cond = self.cond(sinusoidal(sigma.to(h.dtype), self.dim, self.sigma_scale))
        for block in self.blocks:
            h = block(h, cond, manual_attn)
        s, b = self.mod(cond)[:, None].chunk(2, dim=-1)
        return self.head(self.norm(h) * (1 + s) + b)


def build_model(cfg, vocab_size, seq_len, kind="field"):
    """cfg is the `model` section of the config. `kind` is 'field' or 'flow_map'."""
    if cfg.get("arch", "denoiser") == "dit":
        from .dit import DiT
        second = kind == "flow_map" and cfg["cond"] != "duration"   # the target time is a second input
        return DiT(vocab_size, seq_len, cfg["dim"], cfg["depth"], cfg["heads"], cfg["cond_dim"], cfg["dropout"],
                   cfg["softcap"], cfg["logit_cap"], second_input=second, self_cond=cfg.get("self_cond", False))
    if cfg.get("cond", "duration") != "duration":
        raise ValueError("model.arch = denoiser only supports model.cond = duration")
    return Denoiser(vocab_size, seq_len, cfg["dim"], cfg["depth"], cfg["heads"], cfg["mlp_ratio"], cfg["dropout"],
                    cfg["sigma_scale"])
