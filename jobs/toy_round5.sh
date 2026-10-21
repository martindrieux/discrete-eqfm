#!/bin/bash
# Toy round 5: two more seeds of the three key variants (plain, random clean positions, model-chosen clean positions),
# 60k steps, denoisers (D) and flow maps (F1: seed 1, F2: seed 2). Seed 0 is round 4.
cd ~/discrete-eqfm && source jobs/env.sh
export EVAL_OPTS="grid=reparam"
F=configs/toy_markov.yaml; R="model.arch=dit model.cond_dim=64 train.steps=60000 train.save_every=20000"
CH="loss.partial_context=true loss.freeze_context=true loss.context_frac=0.5"
AD="loss.adaptive=binary loss.adaptive_start=2000 loss.adaptive_h=0.9 loss.adaptive_frac=0.5"
specs() { for s in "$@"; do echo "TM5_$P""plain_s$s|$R seed=$s"; echo "TM5_$P""context_half_s$s|$R seed=$s $CH"; echo "TM5_$P""adapt_h90_half_s$s|$R seed=$s $AD"; done; }
if [ "$1" = D ]; then P=denoiser_; mapfile -t S < <(specs 1 2); bash jobs/toy_group.sh $F train_field.py "${S[@]}"
elif [ "$1" = F1 ]; then P=""; mapfile -t S < <(specs 1); bash jobs/toy_group.sh $F train_flow_map.py "${S[@]}"
else P=""; mapfile -t S < <(specs 2); bash jobs/toy_group.sh $F train_flow_map.py "${S[@]}"; fi
