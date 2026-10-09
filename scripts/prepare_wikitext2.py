"""Tokenise WikiText-2 (raw) into blocks of token ids, in the same way as scripts/prepare_lm1b.py does for LM1B.

    python scripts/prepare_wikitext2.py --out data [--block 128]

Writes <out>/wt2_{train,validation,test}.bin and the matching .json for `data.name: blocks`.

Preprocessing: the WikiText detokeniser of flm/dataloader.py, empty lines dropped, bert-base-uncased without special
tokens, a separator after every line, lines concatenated in batches of 1000 and cut into chunks of block - 2 tokens
(the remainder of each batch is dropped), each chunk with a first and a last token; separator, first and last token
are all [CLS] (101), as in the LM1B preprocessing of FLM.
Differences from FLM's own WikiText-2 config: it uses the GPT-2 tokeniser and keeps empty lines; BERT tokens are used
here so that the vocabulary (30522), the time reparameterisation and the evaluation are those of the LM1B study.
"""
import argparse
import itertools
import json
import os
import re

import numpy as np


def wt_detokenizer(string):
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--block", type=int, default=128)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok.encode(tok.sep_token)[0]          # [CLS] for BERT, as in FLM (see prepare_lm1b.py)
    inner = args.block - 2
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "validation", "test"):
        lines = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip()]
        blocks = []
        for i in range(0, len(lines), 1000):
            ids = tok([wt_detokenizer(t) for t in lines[i:i + 1000]], add_special_tokens=False)["input_ids"]
            flat = list(itertools.chain(*[x + [sep] for x in ids]))
            n = (len(flat) // inner) * inner
            blocks += [[sep] + flat[j:j + inner] + [sep] for j in range(0, n, inner)]
        arr = np.asarray(blocks, dtype=np.uint16)
        path = os.path.join(args.out, f"wt2_{split}.bin")
        with open(path, "wb") as f:
            np.save(f, arr)
        with open(path + ".json", "w") as f:
            json.dump({"vocab_size": len(tok), "dtype": "uint16", "tokenizer": args.tokenizer, "block": args.block,
                       "n_blocks": len(arr), "n_lines": len(lines), "format": "npy"}, f)
        print(f"{split}: {len(lines)} lines -> {len(arr)} blocks of {args.block} tokens at {path}", flush=True)


if __name__ == "__main__":
    main()
