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


DATASETS = {"toy": ToyModes, "text": TextFile, "tokens": TokenFile}


def build_dataset(cfg):
    """cfg is the `data` section of the config."""
    name, seq_len = cfg["name"], cfg["seq_len"]
    if name == "toy":
        return ToyModes(seq_len, cfg["vocab_size"], cfg["n_modes"])
    if name in ("text", "tokens"):
        return DATASETS[name](cfg["path"], seq_len)
    raise ValueError(f"unknown dataset '{name}', choose from {sorted(DATASETS)}")
