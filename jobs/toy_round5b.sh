#!/bin/bash
# Toy round 5b: seeds 1 and 2 of two more flow-map variants with model-chosen clean positions (60k steps).
cd ~/discrete-eqfm && source jobs/env.sh
export EVAL_OPTS="grid=reparam"
F=configs/toy_markov.yaml; R="model.arch=dit model.cond_dim=64 train.steps=60000 train.save_every=20000"
AD="loss.adaptive=binary loss.adaptive_start=2000"
bash jobs/toy_group.sh $F train_flow_map.py "TM5_adapt_h90_s1|$R seed=1 $AD loss.adaptive_h=0.9" "TM5_adapt_h90_s2|$R seed=2 $AD loss.adaptive_h=0.9" \
  "TM5_adapt_h70_half_s1|$R seed=1 $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5" "TM5_adapt_h70_half_s2|$R seed=2 $AD loss.adaptive_h=0.7 loss.adaptive_frac=0.5"
