#!/bin/bash
# Re-evaluates toy checkpoints with another time grid:  bash jobs/toy_reeval.sh GRID name1 name2 ...
cd ~/discrete-eqfm && source jobs/env.sh
g=$1; shift
for n in "$@"; do
  ( $PY -u scripts/eval_toy_markov.py $DEQ_ROOT/runs/$n/last.pt out=runs/$n/toy_eval_$g.json grid=$g > runs/$n/toy_eval_$g.txt 2>&1; echo "$n rc=$?" ) &
done
wait
