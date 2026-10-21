#!/bin/bash
# Continuous adaptive-speed sampler on one WikiText-2 denoiser for a list of laws "base beta power" (8 / 32 / 128 evaluations).
#   bash jobs/eval_speed_laws.sh NAME "0.75 4 1" "1.0 2 1" ...
cd ~/discrete-eqfm && source jobs/env.sh
n=$1; shift; ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
for law in "$@"; do set -- $law; o=runs/$n/eval_speed_b$1_beta$2_p$3
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speed base=$1 beta=$2 power=$3 'eval.steps=[8,32,128]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
done
