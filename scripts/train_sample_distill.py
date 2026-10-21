"""One-step map trained on a denoiser's own samples (no data in this stage).

    python scripts/train_sample_distill.py configs/wt2.yaml loss.teacher=<denoiser checkpoint> out_dir=...

The frozen teacher is a time-free denoiser trained with random clean positions. For a noise x_0, its continuous
adaptive-speed sampler (deqfm/sampling.py: sample_speed_clock, `sdist.K` steps, speed sdist.base + sdist.beta g^sdist.power)
returns tokens y(x_0); the sampler is deterministic given x_0. The student, started from the teacher's weights, is the
whole jump of a flow map (duration 1) and is trained by cross-entropy to y(x_0) at the pure-noise state x_0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from deqfm.clocks import build_clock
from deqfm.config import parse_cli
from deqfm.data import build_dataset
from deqfm.model import build_model
from deqfm.sampling import sample_speed_clock
from deqfm.track import nb
from deqfm.trainer import load_model, train


@nb.core
def sample_distill_loss(student_logits, teacher_tokens):
    """Cross-entropy of the one-step map at the noise x_0 to the tokens the teacher's sampler reaches from x_0."""
    return F.cross_entropy(student_logits.float().flatten(0, 1), teacher_tokens.flatten())


def main():
    cfg = parse_cli()
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size, dataset.seq_len, "flow_map")
    teacher, ckpt = load_model(cfg.loss.teacher)
    assert ckpt["kind"] == "field" and ckpt["cfg"]["model"]["cond"] == "duration"
    model.load_state_dict(teacher.state_dict(), strict=False)
    clock, S, V = build_clock(cfg.clock), cfg.sdist, dataset.vocab_size

    def loss_fn(net, raw, tokens):
        B, L = tokens.shape
        x0 = torch.randn(B, L, V, device=tokens.device)
        with torch.no_grad():
            y = sample_speed_clock(teacher, "duration", clock, B, S.K, beta=S.beta, power=S.power, base=S.base,
                                   device=tokens.device, x0=x0)
        loss = sample_distill_loss(net(x0, torch.ones(B, device=tokens.device)), y)
        return loss, {"loss": loss.item()}

    train(cfg, model, dataset, loss_fn, kind="flow_map", teacher=teacher)


if __name__ == "__main__":
    main()
