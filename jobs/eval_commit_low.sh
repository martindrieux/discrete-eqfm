#!/bin/bash
# Commit sampler at low thresholds on WikiText-2 denoisers, to compare with soft continuous speed laws at equal entropy.
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for h in 0.3 0.5 0.7; do o=runs/$n/eval_adaptive_h$h
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=adaptive threshold=$h 'eval.steps=[8,32,128]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
  for law in "0.75 4 1" "1.0 16 2" "1.0 4 1"; do set -- $law; o=runs/$n/eval_speed_b$1_beta$2_p$3
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speed base=$1 beta=$2 power=$3 'eval.steps=[8,32,128]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
