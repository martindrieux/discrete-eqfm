#!/bin/bash
# Entropy / perplexity frontier of a checkpoint: sweep the scale of the initial noise.  bash jobs/frontier.sh TAG CKPT [extra eval args]
tag=$1; ck=$2; shift 2; mkdir -p runs/frontier
for a in 0.9 0.95 1.0 1.03 1.06 1.1 1.15; do
  [ -f runs/frontier/${tag}_a$a.json ] || $PY -u scripts/eval_lm.py $ck out=runs/frontier/${tag}_a$a.json 'eval.steps=[1,2,4]' noise_scale=$a "$@"
done
