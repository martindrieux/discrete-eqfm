"""Sample-quality metrics for language models, computed as in FLM / FMLM (flm/metrics.py):

generative perplexity : samples are decoded to text with the model's tokenizer, re-tokenised for a judge language
                        model (GPT-2 Large) in groups of `group` samples (padded to the longest of the group,
                        truncated to the sequence length), and scored. exp(mean negative log-likelihood per token).
entropy               : mean over samples of the entropy (nats) of the token histogram of each sample.
                        Low values mean repeated tokens, which also lower the perplexity.
"""
import torch
import torch.nn.functional as F

from .sampling import unigram_entropy

_JUDGES = {}


def _judge(name, device):
    if name not in _JUDGES:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _JUDGES[name] = (tok, AutoModelForCausalLM.from_pretrained(name).eval().to(device))
    return _JUDGES[name]


@torch.no_grad()
def generative_perplexity(texts, max_length, judge="gpt2-large", device="cuda", group=16, batch=8):
    """Returns (perplexity, number of scored tokens)."""
    tok, model = _judge(judge, device)
    total, count = 0.0, 0
    for i in range(0, len(texts), group):
        enc = tok(texts[i:i + group], return_tensors="pt", return_token_type_ids=False, return_attention_mask=True,
                  truncation=True, padding=True, max_length=max_length)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        for j in range(0, ids.shape[0], batch):
            x, m = ids[j:j + batch], mask[j:j + batch]
            logits = model(x, attention_mask=m)[0].transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1], x[..., 1:], reduction="none")
            # every token before the padding, plus the first padding (end-of-text) token
            first_eos = (x == tok.eos_token_id).cumsum(-1) == 1
            valid = (first_eos[..., 1:] | (x != tok.eos_token_id)[..., 1:]).double()
            total += (nll.double() * valid).sum().item()
            count += valid.sum().item()
    return float(torch.tensor(total / count).exp()), int(count)


def ngram_keys(tokens, n=8):
    """Every run of n consecutive tokens of every row, as one integer key each. tokens: integer array [N, L]."""
    import numpy as np
    t = np.ascontiguousarray(np.asarray(tokens)).astype(np.uint64)
    key = np.zeros((t.shape[0], t.shape[1] - n + 1), dtype=np.uint64)
    for j in range(n):
        key = key * np.uint64(1000003) + t[:, j:t.shape[1] - n + 1 + j]
    return key.ravel()


def copy_rate(tokens, train_keys, n=8):
    """Fraction of the n-token runs of the samples that occur verbatim in the training set (0: nothing copied)."""
    import numpy as np
    return float(np.isin(ngram_keys(tokens.cpu().numpy(), n), train_keys).mean())


def text_metrics(tokens, tokenizer, judge="gpt2-large", device="cuda", train_keys=None):
    """tokens: LongTensor [N, L] in the model's vocabulary."""
    texts = tokenizer.batch_decode(tokens.cpu())
    ppl, n = generative_perplexity(texts, tokens.shape[1], judge, device)
    t = tokens.cpu()
    out = {"gen_ppl": ppl, "entropy": unigram_entropy(t), "judge_tokens": n, "num": len(texts),
           # doubled tokens: share of positions equal to the previous one, and to the one two back (real text: about 0.1 % and 1 %)
           "dup1": (t[:, 1:] == t[:, :-1]).float().mean().item(), "dup2": (t[:, 2:] == t[:, :-2]).float().mean().item()}
    if train_keys is not None:
        out["copy8"] = copy_rate(tokens, train_keys)
    return out, texts
