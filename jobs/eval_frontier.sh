#!/bin/bash
# Perplexity against entropy for two WikiText-2 denoisers, by scaling the initial noise (32 and 128 steps):
#   plain denoiser with Euler steps, and the denoiser trained with random clean positions sampled with commits (threshold 0.5)
cd ~/discrete-eqfm && source jobs/env.sh
ev() { n=$1; tag=$2; shift 2; [ -f runs/$n/$tag.json ] || { $PY -u scripts/eval_lm.py $DEQ_ROOT/runs/$n/best.pt out=runs/$n/$tag.tmp.json 'eval.steps=[32,128]' "$@" > runs/$n/$tag.log 2>&1 && mv runs/$n/$tag.tmp.json runs/$n/$tag.json; }; }
for s in 0.8 0.9 0.95 1.05 1.1; do
  ev W2_denoiser_timefree frontier_plain_s$s noise_scale=$s
  ev $1 frontier_commit_s$s sampler=adaptive threshold=0.5 noise_scale=$s
  ev $1 frontier_plain_s$s noise_scale=$s
done
