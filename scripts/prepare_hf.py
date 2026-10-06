"""Tokenise a Hugging Face text dataset into a flat binary file of token ids for `data.name: tokens`.

    python scripts/prepare_hf.py --dataset Skylion007/openwebtext --tokenizer gpt2 --out data/owt.bin
    python scripts/prepare_hf.py --dataset wikitext --config wikitext-103-raw-v1 --tokenizer gpt2 --out data/wt103.bin

Writes <out> and <out>.json with {"vocab_size", "dtype", "tokenizer"}. Needs `datasets` and `transformers`.
Documents are concatenated with the tokeniser's end-of-text token between them.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    ds = load_dataset(args.dataset, args.config, split=args.split)

    def encode(batch):
        ids = tok(batch[args.text_column], add_special_tokens=False)["input_ids"]
        ids = [x + [sep] if sep is not None else x for x in ids]
        return {"ids": ids, "len": [len(x) for x in ids]}

    ds = ds.map(encode, batched=True, num_proc=args.num_proc, remove_columns=ds.column_names)
    total = int(np.sum(ds["len"], dtype=np.int64))
    dtype = "uint16" if len(tok) < 2 ** 16 else "uint32"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = np.memmap(args.out, dtype=dtype, mode="w+", shape=(total,))
    pos = 0
    for batch in ds.iter(batch_size=1000):
        flat = np.concatenate([np.asarray(x, dtype=dtype) for x in batch["ids"]])
        out[pos:pos + len(flat)] = flat
        pos += len(flat)
    out.flush()
    with open(args.out + ".json", "w") as f:
        json.dump({"vocab_size": len(tok), "dtype": dtype, "tokenizer": args.tokenizer}, f)
    print(f"wrote {total} tokens to {args.out} (vocab {len(tok)}, {dtype})")


if __name__ == "__main__":
    main()
