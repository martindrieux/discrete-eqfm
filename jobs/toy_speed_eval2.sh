#!/bin/bash
# bash jobs/toy_speed_eval2.sh name1 name2 ...   (smooth against binary speed laws, scripts/eval_toy_speed2.py)
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ( $PY -u scripts/eval_toy_speed2.py $DEQ_ROOT/runs/$n/last.pt out=runs/$n/speed_laws.json > runs/$n/speed_laws.txt 2>&1; echo "$n rc=$?" ) &
done
wait
