#!/bin/bash
# First toy round for Section 7 of the notes (Markov chain). usage: bash jobs/toy_round1.sh A|B   (inside a GPU job)
cd ~/discrete-eqfm && source jobs/env.sh
F=configs/toy_markov.yaml; R="train.resume=false"; AD="loss.adaptive_start=2000"
if [ "$1" = A ]; then
  bash jobs/toy_group.sh $F train_flow_map.py "TM_plain|$R" "TM_context|$R loss.partial_context=true loss.freeze_context=true" \
    "TM_adapt_h90|$R loss.adaptive=binary loss.adaptive_h=0.9 $AD" "TM_adapt_h70|$R loss.adaptive=binary loss.adaptive_h=0.7 $AD" \
    "TM_adapt_h50|$R loss.adaptive=binary loss.adaptive_h=0.5 $AD"
else
  bash jobs/toy_group.sh $F train_flow_map.py "TM_adapt_rollout_h90|$R loss.adaptive=rollout loss.adaptive_h=0.9 $AD" \
    "TM_adapt_rollout_h70|$R loss.adaptive=rollout loss.adaptive_h=0.7 $AD" &
  bash jobs/toy_group.sh $F train_field.py "TM_denoiser_plain|$R" "TM_denoiser_adapt_h90|$R loss.adaptive=binary loss.adaptive_h=0.9 $AD" \
    "TM_denoiser_adapt_h70|$R loss.adaptive=binary loss.adaptive_h=0.7 $AD"
  wait
fi
