#!/bin/bash
# Toy round 6: denoisers trained with one time per position drawn as an adaptive-speed schedule with random
# confidences (no model in the loop), sampled with the adaptive-speed sampler. 60k steps, seed 0.
cd ~/discrete-eqfm && source jobs/env.sh
export EVAL_OPTS="grid=reparam beta=1,4,16,64"
F=configs/toy_markov.yaml; R="model.arch=dit model.cond_dim=64 train.steps=60000 train.save_every=20000"
bash jobs/toy_group.sh $F train_field.py "TM6_denoiser_randspeed_b4|$R loss.random_speed=4" "TM6_denoiser_randspeed_b16|$R loss.random_speed=16" \
  "TM6_denoiser_randspeed_b64|$R loss.random_speed=64" "TM6_denoiser_randspeed_b16_half|$R loss.random_speed=16 loss.context_frac=0.5" \
  "TM6_denoiser_adapt_h90_all|$R loss.adaptive=binary loss.adaptive_start=2000 loss.adaptive_h=0.9"
