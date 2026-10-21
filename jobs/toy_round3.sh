#!/bin/bash
# Toy round 3 (Markov chain, network with rotary positions so that the plain model learns to use the other positions).
#   A: flow maps (semigroup loss)   B: denoisers        usage: sbatch jobs/gpu.sbatch "bash jobs/toy_round3.sh A"
cd ~/discrete-eqfm && source jobs/env.sh
F=configs/toy_markov.yaml; R="train.resume=false model.arch=dit model.cond_dim=64"; AD="loss.adaptive=binary loss.adaptive_start=2000"
if [ "$1" = A ]; then
  bash jobs/toy_group.sh $F train_flow_map.py "TM3_plain|$R" "TM3_context|$R loss.partial_context=true loss.freeze_context=true" \
    "TM3_adapt_h90|$R $AD loss.adaptive_h=0.9" "TM3_adapt_h70|$R $AD loss.adaptive_h=0.7" \
    "TM3_adapt_h90_half|$R $AD loss.adaptive_h=0.9 loss.adaptive_frac=0.5" "TM3_adapt_h70_half|$R $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5"
elif [ "$1" = B ]; then
  bash jobs/toy_group.sh $F train_field.py "TM3_denoiser_plain|$R" "TM3_denoiser_context|$R loss.partial_context=true loss.freeze_context=true" \
    "TM3_denoiser_adapt_h90|$R $AD loss.adaptive_h=0.9" "TM3_denoiser_adapt_h70|$R $AD loss.adaptive_h=0.7" \
    "TM3_denoiser_adapt_h90_half|$R $AD loss.adaptive_h=0.9 loss.adaptive_frac=0.5" "TM3_denoiser_adapt_h70_half|$R $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5"
else
  bash jobs/toy_group.sh $F train_flow_map.py "TM3_adapt_rollout_h90|$R loss.adaptive=rollout loss.adaptive_start=2000 loss.adaptive_h=0.9" \
    "TM3_adapt_rollout_h90_half|$R loss.adaptive=rollout loss.adaptive_start=2000 loss.adaptive_h=0.9 loss.adaptive_frac=0.5" \
    "TM3_plain_seed1|$R seed=1" "TM3_context_half|$R loss.partial_context=true loss.freeze_context=true loss.context_frac=0.5"
fi
