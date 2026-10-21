#!/bin/bash
# Continuous adaptive-speed sampler on WikiText-2 denoisers (sharp speed laws), 8 / 32 / 128 evaluations.
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ck=$DEQ_ROOT/runs/$n/best.pt
  for law in "0.75 1024 16" "1.0 1024 16" "0.75 64 8" "0.75 16384 32" "0.75 64 4" "0.75 16 2"; do
    set -- $law; o=runs/$n/eval_speed_b$1_beta$2_p$3
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speed base=$1 beta=$2 power=$3 'eval.steps=[8,32,128]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
