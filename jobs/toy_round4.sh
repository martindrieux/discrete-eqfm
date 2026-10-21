#!/bin/bash
# Toy round 4 (Markov chain, rotary network, samplers on the decoding-clock grid).
#   A: denoisers, continuous adaptive speed (one time per position), 20k steps
#   B: denoisers, 60k steps: does the gap between plain and committed states survive longer training?
#   C: flow maps, 60k steps
cd ~/discrete-eqfm && source jobs/env.sh
export EVAL_OPTS="grid=reparam"
F=configs/toy_markov.yaml; R="model.arch=dit model.cond_dim=64"; AD="loss.adaptive_start=2000"
SP="loss.adaptive=speed loss.adaptive_rollout_steps=8 $AD"
if [ "$1" = A ]; then
  bash jobs/toy_group.sh $F train_field.py "TM4_denoiser_speed_b4|$R $SP loss.adaptive_beta=4" "TM4_denoiser_speed_b16|$R $SP loss.adaptive_beta=16" \
    "TM4_denoiser_speed_b4_half|$R $SP loss.adaptive_beta=4 loss.adaptive_frac=0.5" "TM4_denoiser_speed_b16_half|$R $SP loss.adaptive_beta=16 loss.adaptive_frac=0.5" \
    "TM4_denoiser_speed_b1|$R $SP loss.adaptive_beta=1"
elif [ "$1" = B ]; then
  L="train.steps=60000 train.save_every=20000"
  bash jobs/toy_group.sh $F train_field.py "TM4_long_denoiser_plain|$R $L" "TM4_long_denoiser_context|$R $L loss.partial_context=true loss.freeze_context=true" \
    "TM4_long_denoiser_context_half|$R $L loss.partial_context=true loss.freeze_context=true loss.context_frac=0.5" \
    "TM4_long_denoiser_adapt_h90_half|$R $L loss.adaptive=binary $AD loss.adaptive_h=0.9 loss.adaptive_frac=0.5" \
    "TM4_long_denoiser_adapt_h70_half|$R $L loss.adaptive=binary $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5" \
    "TM4_long_denoiser_plain_wide|$R $L model.dim=256 model.depth=6 train.lr=3e-4"
else
  L="train.steps=60000 train.save_every=20000"
  bash jobs/toy_group.sh $F train_flow_map.py "TM4_long_plain|$R $L" "TM4_long_context_half|$R $L loss.partial_context=true loss.freeze_context=true loss.context_frac=0.5" \
    "TM4_long_adapt_h90|$R $L loss.adaptive=binary $AD loss.adaptive_h=0.9" "TM4_long_adapt_h70_half|$R $L loss.adaptive=binary $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5" \
    "TM4_long_adapt_h90_half|$R $L loss.adaptive=binary $AD loss.adaptive_h=0.9 loss.adaptive_frac=0.5"
fi
