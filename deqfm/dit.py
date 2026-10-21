"""The diffusion transformer of FLM / FMLM (github.com/david3684/flm, models/dit.py), without the flash-attn dependency.

Parameter names match the original, so the released FLM and FMLM checkpoints load with `load_state_dict`.

    forward(x, a, b=None) -> logits [B, L, V]

x is a noisy one-hot state [B, L, V]. `a` and `b` are scalars per sample, embedded and added (`sigma_map`,
`sigma_map_prime`); what they mean is decided by the caller (deqfm/nets.py):

    time-free flow map        a = duration sigma
    FLM denoiser              a = reparameterised time tau
    FMLM two-time denoiser    a = tau of the state, b = tau of the target time

Differences from the original, none of which change the function computed: attention is a plain matmul with the same
tanh soft-cap on the attention scores (so it also supports forward-mode differentiation), and the 30 tanh(logits / 30)
cap that FLM applies outside the backbone is applied here (`logit_cap`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[..., None].float() * freqs       # t: [B], or [B, L] for one value per position
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def rotary_cos_sin(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)   # [L, head_dim / 2]
    return freqs.cos(), freqs.sin()


def apply_rotary(x, cos, sin):
    """x: [B, H, L, D]. Same rotation as flash_attn.layers.rotary.apply_rotary_emb_torch (halves, not interleaved)."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, softcap=50.0):
        super().__init__()
        self.n_heads, self.softcap, self.dropout = n_heads, softcap, dropout
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate='tanh'),
                                 nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, cos, sin, c):
        B, L, C = x.shape
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (mod if mod.dim() == 3 else mod[:, None]).chunk(6, dim=2)
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        q, k, v = self.attn_qkv(h).view(B, L, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, cos.to(q.dtype), sin.to(q.dtype)), apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        h = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, C)
        x = x + gate_msa * F.dropout(self.attn_out(h), p=self.dropout, training=self.training)
        h = self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x + gate_mlp * F.dropout(h, p=self.dropout, training=self.training)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return (x.float() @ self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        for layer in (self.linear, self.adaLN_modulation):
            layer.weight.data.zero_()
            layer.bias.data.zero_()

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)
        shift, scale = (mod if mod.dim() == 3 else mod[:, None]).chunk(2, dim=2)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class DiT(nn.Module):
    def __init__(self, vocab_size, seq_len, dim=768, depth=12, heads=12, cond_dim=128, dropout=0.1, softcap=50.0,
                 logit_cap=30.0, second_input=False, mixed_precision=True, self_cond=False):
        super().__init__()
        self.vocab_size, self.seq_len, self.dim, self.heads = vocab_size, seq_len, dim, heads
        self.logit_cap, self.mixed_precision = logit_cap, mixed_precision
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        # self-conditioning (ELF, arXiv 2605.10938): a draft p of the clean tokens, embedded and added to the input;
        # starts as a no-op, and p = 0 (or None) means no draft
        self.sc_embed = EmbeddingLayer(dim, vocab_size) if self_cond else None
        if self_cond:
            self.sc_embed.embedding.data.zero_()
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.sigma_map_prime = TimestepEmbedder(cond_dim) if second_input else None
        if second_input:   # starts as a no-op, as when FMLM is initialised from FLM
            self.sigma_map_prime.mlp[2].weight.data.zero_()
            self.sigma_map_prime.mlp[2].bias.data.zero_()
        self.blocks = nn.ModuleList(DDiTBlock(dim, heads, cond_dim, dropout=dropout, softcap=softcap)
                                    for _ in range(depth))
        self.output_layer = DDiTFinalLayer(dim, vocab_size, cond_dim)

    def forward(self, x, a, b=None, manual_attn=False, sc=None):
        """x: [B, L, V] float; a, b: [B], or [B, L] for one value per position; sc: [B, L, V] draft (self-conditioning).
        Returns logits [B, L, V]. (`manual_attn` is accepted and ignored.)"""
        h = self.vocab_embed(x)
        if sc is not None and self.sc_embed is not None:
            h = h + self.sc_embed(sc)
        emb = self.sigma_map(a)
        if b is not None:
            emb = emb + self.sigma_map_prime(b)
        c = F.silu(emb)
        cos, sin = rotary_cos_sin(h.shape[1], self.dim // self.heads, h.device)
        amp = self.mixed_precision and h.is_cuda
        with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16, enabled=amp):
            for block in self.blocks:
                h = block(h, cos, sin, c)
            out = self.output_layer(h, c)
        out = out.float()
        if self.logit_cap > 0:
            out = self.logit_cap * torch.tanh(out / self.logit_cap)
        return out
