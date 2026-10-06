# discrete-eqfm

PyTorch training code for equilibrium flow maps on discrete data (token sequences), with a softmax
output and cross-entropy losses.

## What it trains

A sequence of `L` tokens from a vocabulary of size `V` is a one-hot matrix $x_1 \in \mathbb{R}^{L\times V}$.
States are $I_t = (1-t)x_0 + t x_1$ with $x_0 \sim \mathsf{N}(0, I)$. The network has **no time input**.

**1. Autonomous field** (`scripts/train_field.py`). A softmax network $\hat D(x)$ trained with

$$\mathcal{L} = \mathbb{E}\Big[-\frac{1}{1-t} x_1 \cdot \log \hat D(I_t)\Big], \qquad t \sim \mathsf{U}[0,1].$$

The field is $\hat D(x) - x$. Sampling integrates $\dot x = \hat D(x) - x$.

**2. Equilibrium flow map** (`scripts/train_flow_map.py`). A softmax network $\hat\delta_\sigma(x)$ with a
duration input $\sigma \in [0,1]$. One jump is

$$\hat\Phi_\sigma(x) = (1-\kappa(\sigma)) x + \kappa(\sigma) \hat\delta_\sigma(x),$$

where $\kappa$ comes from the clock ($\kappa(\sigma) = \sigma$ for the exponential clock). The loss is
`diag + alpha * off`:

- **diag** ($\sigma = 0$): the field loss above applied to $\hat\delta_0$, or, for distillation, the
  cross-entropy to a frozen field $\hat D$.
- **off** ($\sigma > 0$): cross-entropy $-\bar\delta \cdot \log \hat\delta_\sigma(x)$ to a stop-gradient target
  $\bar\delta$, chosen with `loss.off_diagonal`:

| `off_diagonal` | target $\bar\delta$ | extra network evaluations |
|---|---|---|
| `semigroup` | $\gamma \hat\delta_{\sigma_1}(x) + (1-\gamma) \hat\delta_{\sigma_2}(x')$, with $x' = \hat\Phi_{\sigma_1}(x)$ | 2 |
| `lagrangian` | $\bar D(x') - c \kappa \partial_\sigma\hat\delta_\sigma(x)$, with $x' = \hat\Phi_\sigma(x)$ | 1 JVP + 1 |
| `eulerian` | $(1-\kappa)\bar D(x) + \kappa \hat\delta_\sigma(x) + \kappa J$, with $J$ the derivative of $\hat\delta$ along $(\bar D(x) - x, -c)$ | 1 JVP + 1 |

$\bar D$ is the model's own $\hat\delta_0$ (training from data) or the frozen field (distillation), and
$c = c(\sigma)$ is the clock speed. The code follows Algorithms 1 to 5 of the accompanying notes line by line;
see `deqfm/losses.py` and `deqfm/objectives.py`.

## Install

```bash
pip install -r requirements.txt
```

## Quick start (toy data, CPU)

```bash
# 1. the autonomous field
python scripts/train_field.py configs/toy.yaml out_dir=runs/toy_field
python scripts/sample.py runs/toy_field/last.pt sample.steps=64

# 2a. a flow map trained from data
python scripts/train_flow_map.py configs/toy.yaml loss.off_diagonal=semigroup out_dir=runs/toy_semigroup

# 2b. a flow map distilled from the field
python scripts/train_flow_map.py configs/toy.yaml loss.off_diagonal=eulerian \
       loss.teacher=runs/toy_field/last.pt out_dir=runs/toy_eulerian
python scripts/sample.py runs/toy_eulerian/last.pt sample.steps=1
```

Any config value can be overridden on the command line as `section.key=value`.
For several GPUs, replace `python` by `accelerate launch`.

## Datasets

A dataset yields token ids of shape `[seq_len]` and exposes `vocab_size`, `seq_len` and `decode`.

| `data.name` | source | notes |
|---|---|---|
| `toy` | synthetic | a few fixed sequences with Zipf probabilities; `valid_rate` counts samples that are one of them |
| `text` | a plain text file (`data.path`) | character level, random crops (for example text8) |
| `tokens` | a flat binary file of token ids (`data.path`, plus `data.path + ".json"`) | random crops through a memory map; any tokeniser |

`scripts/prepare_hf.py` writes a `tokens` file from any Hugging Face text dataset and tokeniser:

```bash
python scripts/prepare_hf.py --dataset Skylion007/openwebtext --tokenizer gpt2 --out data/owt.bin
```

To add a dataset, write a class with the three members above and register it in `deqfm/data.py`.

## Configuration

Defaults are in `deqfm/config.py`; a YAML file only lists what differs.

| key | default | meaning |
|---|---|---|
| `clock` | `exponential` | `exponential` ($\kappa = \sigma$, $c = 1-\sigma$) or `rational` ($r_\tau = \tau/(1+\tau)$) |
| `model.dim / depth / heads` | 256 / 4 / 4 | transformer size |
| `model.sigma_scale` | 100 | largest frequency of the duration embedding |
| `loss.diag_time` | `uniform` | `uniform`: $t \sim \mathsf{U}[0, 1-\epsilon]$ with weight $\frac{1}{1-t}$. `log_uniform`: density $\propto \frac{1}{1-t}$, constant weight (same objective, lower variance) |
| `loss.t_eps` | 1e-4 | $\epsilon$ above: a numerical guard on $t$ near 1 |
| `loss.off_diagonal` | `semigroup` | `semigroup`, `lagrangian` or `eulerian` |
| `loss.alpha` | 1.0 | weight of the off-diagonal loss |
| `loss.jvp` | `forward` | `forward` (`torch.func.jvp`) or `fd` (central finite differences, step `loss.fd_eps`) |
| `loss.split_batch` | true | half of each batch for each loss term |
| `loss.teacher` | none | path to a field checkpoint: switches to distillation |
| `loss.init_from_teacher` | true | start the flow map from the teacher weights |
| `train.batch_size` | 64 | per process |
| `train.grad_accum` | 1 | gradient accumulation steps |
| `train.lr / warmup` | 3e-4 / 2500 | AdamW, linear warmup then constant |
| `train.ema` | 0.9999 | EMA decay; samples and teachers use the EMA weights |
| `train.precision` | `no` | `no`, `fp16` or `bf16` |
| `train.resume` | true | continue from `out_dir/last.pt` if present |

Durations are sampled as in the notes: $\sigma \sim \mathsf{U}[0,1]$ for the Lagrangian and Eulerian losses;
for the semigroup loss $\sigma_3$ has density $2\sigma_3$, $\sigma_1 \sim \mathsf{U}[0, \sigma_3]$, and $\sigma_2$
is set so that the two jumps compose to $\sigma_3$. Losses are averaged over positions.

## Scaling

- One state is `batch x seq_len x vocab` numbers, the same size as the logits. For large vocabularies keep
  the per-process batch small and use several GPUs and `train.grad_accum`.
- Forward-mode differentiation does not go through PyTorch's fused attention kernel, so targets that need a
  Jacobian-vector product use a plain matmul attention (`manual_attn=True`). `loss.jvp=fd` avoids this.
- Mixed precision, multi-GPU and gradient accumulation are handled by `accelerate`.

## Layout

```
deqfm/clocks.py       c(sigma), kappa(sigma), kappa^{-1} for each clock
deqfm/data.py         datasets
deqfm/model.py        transformer denoiser with a duration input
deqfm/losses.py       state sampling, diagonal loss, the three off-diagonal targets, JVPs
deqfm/objectives.py   field loss and flow-map loss (Algorithm 1)
deqfm/sampling.py     sampler (Algorithm 5)
deqfm/trainer.py      training loop, EMA, checkpoints, logging
scripts/              train_field.py, train_flow_map.py, sample.py, prepare_hf.py
configs/              toy.yaml, text8.yaml, owt.yaml
tests/test_smoke.py   clock identities, JVP against finite differences, targets, a few training steps
```

## Status

See `RESULTS.md` for what has been run.
