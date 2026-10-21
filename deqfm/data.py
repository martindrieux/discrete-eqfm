"""Datasets. Every dataset yields LongTensors of token ids with shape [seq_len] and exposes

    vocab_size : int
    seq_len    : int
    decode(tokens) -> str

To add a dataset, write a class with these three members and register it in DATASETS.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_NOMINAL_LEN = 1_000_000  # random-crop datasets have no natural epoch


class ToyModes(Dataset):
    """A finite set of `n_modes` fixed random sequences with Zipf probabilities.

    Every position is strongly correlated with every other one, so a sampler that treats
    positions independently produces sequences outside the set. `valid_rate` measures this.
    """

    def __init__(self, seq_len=16, vocab_size=16, n_modes=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.modes = torch.randint(0, vocab_size, (n_modes, seq_len), generator=g)
        p = 1.0 / torch.arange(1, n_modes + 1)
        self.probs = p / p.sum()
        self.vocab_size, self.seq_len = vocab_size, seq_len

    def __len__(self):
        return _NOMINAL_LEN

    def __getitem__(self, i):
        return self.modes[torch.multinomial(self.probs, 1).item()]

    def decode(self, tokens):
        return " ".join(str(int(t)) for t in tokens)

    def valid_rate(self, samples):
        """Fraction of sampled sequences [N, seq_len] that are one of the modes."""
        hit = (samples[:, None, :].cpu() == self.modes[None]).all(-1).any(-1)
        return hit.float().mean().item()

    def mode_tv(self, samples):
        """Total variation distance between the mode frequencies among valid samples and the true probabilities."""
        match = (samples[:, None, :].cpu() == self.modes[None]).all(-1).float()   # [N, n_modes]
        counts = match.sum(0)
        if counts.sum() == 0:
            return 1.0
        return 0.5 * (counts / counts.sum() - self.probs).abs().sum().item()


class ToyMarkov(Dataset):
    """Sequences of a sparse Markov chain: each token has `branching` allowed successors, with random probabilities.

    There are vocab_size * branching^(seq_len - 1) valid sequences, so nothing can be memorised, and every metric is
    exact: a sample is valid if all its transitions are allowed, its likelihood under the chain is known, and the
    frequencies of its token pairs can be compared with the true ones.
    """

    def __init__(self, seq_len=32, vocab_size=64, branching=4, seed=0, pool=1_000_000):
        g = torch.Generator().manual_seed(seed)
        P = torch.zeros(vocab_size, vocab_size)
        for i in range(vocab_size):
            nxt = torch.randperm(vocab_size, generator=g)[:branching]
            w = torch.rand(branching, generator=g) + 0.2
            P[i, nxt] = w / w.sum()
        self.P = P
        pi = torch.full((vocab_size,), 1.0 / vocab_size)
        for _ in range(2000):                       # stationary distribution; sequences start from it
            pi = pi @ P
        self.pi = pi / pi.sum()
        self.vocab_size, self.seq_len = vocab_size, seq_len
        self.pair = self.pi[:, None] * P            # true frequency of each pair of adjacent tokens
        logP = torch.where(P > 0, P.clamp_min(1e-30).log(), torch.zeros_like(P))
        self.entropy_rate = -(self.pair * logP).sum().item()     # nats per token: the likelihood of real sequences
        self.pool = self.draw(pool, torch.Generator().manual_seed(seed + 1))    # training sequences, drawn once

    def __len__(self):
        return len(self.pool)

    def draw(self, n, generator=None):
        x = torch.empty(n, self.seq_len, dtype=torch.long)
        x[:, 0] = torch.multinomial(self.pi, n, replacement=True, generator=generator)
        for l in range(1, self.seq_len):
            x[:, l] = torch.multinomial(self.P[x[:, l - 1]], 1, generator=generator).squeeze(-1)
        return x

    def __getitem__(self, i):
        return self.pool[i]

    def decode(self, tokens):
        return " ".join(str(int(t)) for t in tokens)

    def metrics(self, samples):
        """valid: fraction of sequences whose transitions are all allowed. step_valid: fraction of allowed transitions.
        nll: mean negative log-likelihood per transition under the chain, over allowed transitions (the entropy rate
        for real data; lower means the samples keep to the likeliest successors). pair_tv: total variation distance
        between the pair frequencies of the samples and the true ones (0 is perfect; high when samples lack variety)."""
        s = samples.cpu()
        a, b = s[:, :-1], s[:, 1:]
        p = self.P[a, b]
        ok = p > 0
        counts = torch.zeros(self.vocab_size * self.vocab_size)
        counts.scatter_add_(0, (a * self.vocab_size + b).flatten(), torch.ones(a.numel()))
        return {"valid": ok.all(-1).float().mean().item(), "step_valid": ok.float().mean().item(),
                "nll": (-(p[ok].log()).mean().item() if ok.any() else float("nan")),
                "pair_tv": 0.5 * (counts / counts.sum() - self.pair.flatten()).abs().sum().item(),
                "distinct": len({tuple(r.tolist()) for r in s}) / len(s)}


class TextFile(Dataset):
    """Character-level random crops of a plain text file (for example text8)."""

    def __init__(self, path, seq_len=256):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(set(text))
        lookup = {ch: i for i, ch in enumerate(self.chars)}
        self.data = torch.tensor([lookup[ch] for ch in text], dtype=torch.long)
        self.vocab_size, self.seq_len = len(self.chars), seq_len

    def __len__(self):
        return _NOMINAL_LEN

    def __getitem__(self, i):
        start = torch.randint(0, len(self.data) - self.seq_len, (1,)).item()
        return self.data[start:start + self.seq_len]

    def decode(self, tokens):
        return "".join(self.chars[int(t)] for t in tokens)


class TokenFile(Dataset):
    """Random crops of a flat binary file of token ids, read through a memory map.

    `path` is the .bin file; `path + '.json'` holds {"vocab_size", "dtype", "tokenizer"}.
    Create both with scripts/prepare_hf.py, or write them yourself for any tokeniser.
    """

    def __init__(self, path, seq_len=1024):
        with open(path + ".json") as f:
            self.meta = json.load(f)
        self.path, self.seq_len = path, seq_len
        self.vocab_size = int(self.meta["vocab_size"])
        self.n_tokens = os.path.getsize(path) // np.dtype(self.meta["dtype"]).itemsize
        self._data, self._tok = None, None

    def __len__(self):
        return _NOMINAL_LEN

    def __getitem__(self, i):
        if self._data is None:  # opened lazily so that each DataLoader worker has its own handle
            self._data = np.memmap(self.path, dtype=self.meta["dtype"], mode="r")
        start = torch.randint(0, self.n_tokens - self.seq_len, (1,)).item()
        return torch.from_numpy(self._data[start:start + self.seq_len].astype(np.int64))

    def decode(self, tokens):
        name = self.meta.get("tokenizer")
        if name is None:
            return " ".join(str(int(t)) for t in tokens)
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(name)
        return self._tok.decode([int(t) for t in tokens])


class BlockFile(Dataset):
    """Pre-cut sequences: a .npy-format array of token ids [N, seq_len], read through a memory map.

    `path + '.json'` holds {"vocab_size", "tokenizer"}. Written by scripts/prepare_lm1b.py.
    """

    def __init__(self, path, seq_len):
        with open(path + ".json") as f:
            self.meta = json.load(f)
        self.path, self.vocab_size = path, int(self.meta["vocab_size"])
        shape = np.load(path, mmap_mode="r").shape
        assert shape[1] == seq_len, f"{path} has sequences of length {shape[1]}, not {seq_len}"
        self.n, self.seq_len = shape[0], seq_len
        self._data, self._tok = None, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._data is None:
            self._data = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(self._data[i].astype(np.int64))

    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.meta["tokenizer"])
        return self._tok

    def decode(self, tokens):
        return self.tokenizer().decode([int(t) for t in tokens])


DATASETS = {"toy": ToyModes, "markov": ToyMarkov, "text": TextFile, "tokens": TokenFile, "blocks": BlockFile}


def build_dataset(cfg):
    """cfg is the `data` section of the config."""
    name, seq_len = cfg["name"], cfg["seq_len"]
    if name == "toy":
        return ToyModes(seq_len, cfg["vocab_size"], cfg["n_modes"])
    if name == "markov":
        return ToyMarkov(seq_len, cfg["vocab_size"], cfg.get("branching", 4))
    if name in ("text", "tokens", "blocks"):
        return DATASETS[name](cfg["path"], seq_len)
    raise ValueError(f"unknown dataset '{name}', choose from {sorted(DATASETS)}")
