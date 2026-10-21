#!/bin/bash
# Speed-law sweep of the continuous adaptive-speed sampler on toy denoisers:  bash jobs/toy_speed_eval.sh "opts" name1 name2 ...
cd ~/discrete-eqfm && source jobs/env.sh
opts=$1; shift
for n in "$@"; do
  ( $PY -u scripts/eval_toy_speed.py $DEQ_ROOT/runs/$n/last.pt out=runs/$n/speed_sweep${TAG}.json $opts > runs/$n/speed_sweep${TAG}.txt 2>&1; echo "$n rc=$?" ) &
done
wait
