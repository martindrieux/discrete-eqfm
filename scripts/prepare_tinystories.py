"""Tokenise TinyStories as FMLM+ does (github.com/MananAg007/posterior-refinement, data=tinystories, wrap=False).

    python scripts/prepare_tinystories.py --out /data/user_data/$USER/deqfm/data [--block 128]

Writes <out>/tinystories_{train,validation}.bin (uint16, shape [N, block]) and the .json for `data.name: blocks`.
One story per row: <|endoftext|> story <|endoftext|>, GPT-2 tokens, truncated to `block` tokens (the first and the last
stay <|endoftext|>) and padded on the right with <|endoftext|>.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--num_proc", type=int, default=6)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    eos, inner = tok.eos_token_id, args.block - 2

    def encode(batch):
        ids = tok(batch["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        rows = [[eos] + x[:inner] + [eos] for x in ids]
        return {"input_ids": [r + [eos] * (args.block - len(r)) for r in rows], "n": [len(x) for x in ids]}

    os.makedirs(args.out, exist_ok=True)
    for split in ("validation", "train"):
        ds = load_dataset(args.dataset, split=split)
        ds = ds.map(encode, batched=True, batch_size=1000, num_proc=args.num_proc, remove_columns=ds.column_names)
        path = os.path.join(args.out, f"tinystories_{split}.bin")
        out = np.lib.format.open_memmap(path + ".npy", mode="w+", dtype=np.uint16, shape=(len(ds), args.block))
        pos, short = 0, 0
        for batch in ds.iter(batch_size=100_000):
            arr = np.asarray(batch["input_ids"], dtype=np.uint16)
            out[pos:pos + len(arr)] = arr
            pos += len(arr)
            short += int((np.asarray(batch["n"]) < inner).sum())
        out.flush()
        del out
        os.replace(path + ".npy", path)
        with open(path + ".json", "w") as f:
            json.dump({"vocab_size": len(tok), "dtype": "uint16", "tokenizer": "gpt2", "block": args.block,
                       "n_blocks": pos, "n_padded": short, "format": "npy"}, f)
        print(f"{split}: {pos} stories, {short} shorter than {inner} tokens (padded) -> {path}", flush=True)


if __name__ == "__main__":
    main()
