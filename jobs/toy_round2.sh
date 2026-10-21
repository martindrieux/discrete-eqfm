#!/bin/bash
# Toy round 2: can the plain denoiser leave the plateau where it ignores the other positions? (inside a GPU job)
cd ~/discrete-eqfm && source jobs/env.sh
F=configs/toy_markov.yaml; R="train.resume=false"; D="model.arch=dit model.cond_dim=64"
bash jobs/toy_group.sh $F train_field.py "TM2_denoiser_rotary|$R $D" "TM2_denoiser_rotary_lr3|$R $D train.lr=3e-4" \
  "TM2_denoiser_rotary_wide|$R $D model.dim=256 model.depth=4 train.lr=3e-4" "TM2_denoiser_abs_lr3_long|$R train.lr=3e-4 train.steps=60000" \
  "TM2_denoiser_rotary_adapt_h90|$R $D loss.adaptive=binary loss.adaptive_h=0.9 loss.adaptive_start=2000" \
  "TM2_denoiser_rotary_context|$R $D loss.partial_context=true loss.freeze_context=true"
