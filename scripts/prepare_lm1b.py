"""Tokenise LM1B exactly as FLM/FMLM do (github.com/david3684/flm, data=lm1b-wrap) into blocks of token ids.

    python scripts/prepare_lm1b.py --out /data/user_data/$USER/deqfm/data [--block 128]

Writes <out>/lm1b_{train,test}.bin (uint16, shape [N, block]) and the matching .json for `data.name: blocks`.

Preprocessing, following flm/dataloader.py:
  - text goes through `lm1b_detokenizer`, then bert-base-uncased without special tokens;
  - every sentence is followed by a separator; sentences are concatenated in map batches of 1000 and cut into
    chunks of block - 2 tokens (the remainder of each batch is dropped); each chunk gets a first and a last token;
  - separator, first and last token are all `tokenizer.encode(tokenizer.sep_token)[0]`, which for BERT is [CLS] (101),
    because `encode` prepends [CLS]. This is what FLM does, and it is why its samples contain [CLS] between sentences.
"""
import argparse
import itertools
import json
import os
import re

import numpy as np


def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="dvruette/lm1b")   # parquet copy of billion-word-benchmark/lm1b
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--num_proc", type=int, default=len(os.sched_getaffinity(0)))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos = tok.encode(tok.cls_token)[0]
    eos = tok.encode(tok.sep_token)[0]
    print(f"first token {bos} ({tok.convert_ids_to_tokens(bos)}), separator/last token {eos} "
          f"({tok.convert_ids_to_tokens(eos)})", flush=True)
    inner = args.block - 2

    def encode(batch):
        text = [lm1b_detokenizer(t) for t in batch["text"]]
        ids = tok(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        flat = list(itertools.chain(*[x + [eos] for x in ids]))
        n = (len(flat) // inner) * inner
        return {"input_ids": [[bos] + flat[i:i + inner] + [eos] for i in range(0, n, inner)]}

    os.makedirs(args.out, exist_ok=True)
    for split in ("test", "train"):
        ds = load_dataset(args.dataset, split=split)
        n_sent = len(ds)
        ds = ds.map(encode, batched=True, batch_size=1000, num_proc=args.num_proc, remove_columns=ds.column_names,
                    desc=f"tokenising {split}")
        path = os.path.join(args.out, f"lm1b_{split}.bin")
        out = np.lib.format.open_memmap(path + ".npy", mode="w+", dtype=np.uint16, shape=(len(ds), args.block))
        pos = 0
        for batch in ds.iter(batch_size=100_000):
            arr = np.asarray(batch["input_ids"], dtype=np.uint16)
            out[pos:pos + len(arr)] = arr
            pos += len(arr)
        out.flush()
        del out
        os.replace(path + ".npy", path)   # .npy file: 128-byte header, then the raw array
        with open(path + ".json", "w") as f:
            json.dump({"vocab_size": len(tok), "dtype": "uint16", "tokenizer": args.tokenizer, "block": args.block,
                       "n_blocks": pos, "n_sentences": n_sent, "format": "npy"}, f)
        print(f"{split}: {n_sent} sentences -> {pos} blocks of {args.block} tokens at {path}", flush=True)


if __name__ == "__main__":
    main()
