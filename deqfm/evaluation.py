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


def text_metrics(tokens, tokenizer, judge="gpt2-large", device="cuda"):
    """tokens: LongTensor [N, L] in the model's vocabulary."""
    texts = tokenizer.batch_decode(tokens.cpu())
    ppl, n = generative_perplexity(texts, tokens.shape[1], judge, device)
    return {"gen_ppl": ppl, "entropy": unigram_entropy(tokens.cpu()), "judge_tokens": n, "num": len(texts)}, texts
