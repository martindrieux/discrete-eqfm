#!/bin/bash
# Toy round 7: denoisers trained on the states of the continuous adaptive-speed process in the decoding-error clock
# (model in the loop, speed law base + beta * confidence^power), 60k steps, then the speed-law sweep of the sampler.
#   usage: bash jobs/toy_round7.sh A|B
cd ~/discrete-eqfm && source jobs/env.sh
F=configs/toy_markov.yaml; R="model.arch=dit model.cond_dim=64 train.steps=60000 train.save_every=20000"
S="loss.adaptive=speed_clock loss.adaptive_start=2000 loss.adaptive_rollout_steps=8"
MIX="loss.adaptive_frac=0.5 loss.partial_context=true loss.freeze_context=true"
run() {   # name, overrides: train, then sweep the sampler
  local n=$1; shift
  ( [ -f runs/$n/speed_sweep.json ] && exit 0; mkdir -p runs/$n
    $PY -u scripts/train_field.py $F $R $S "$@" out_dir=runs/$n ckpt_dir=$DEQ_ROOT/runs/$n > runs/$n/train.log 2>&1 &&
    $PY -u scripts/eval_toy_speed.py $DEQ_ROOT/runs/$n/last.pt out=runs/$n/speed_sweep.json beta=16,64,256,1024 power=4,8,16 > runs/$n/speed_sweep.txt 2>&1
    echo "$n rc=$?" ) &
}
if [ "$1" = A ]; then
  run TM7_speed_b256_p16_all loss.adaptive_beta=256 loss.adaptive_power=16
  run TM7_speed_b256_p16_half loss.adaptive_beta=256 loss.adaptive_power=16 loss.adaptive_frac=0.5
  run TM7_speed_b256_p16_mix loss.adaptive_beta=256 loss.adaptive_power=16 $MIX
else
  run TM7_speed_b64_p8_mix loss.adaptive_beta=64 loss.adaptive_power=8 $MIX
  run TM7_speed_b16_p4_mix loss.adaptive_beta=16 loss.adaptive_power=4 $MIX
  run TM7_speed_b64_p8_all loss.adaptive_beta=64 loss.adaptive_power=8
fi
wait
