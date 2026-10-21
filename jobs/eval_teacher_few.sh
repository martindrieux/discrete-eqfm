#!/bin/bash
# The teacher denoisers at 2 / 4 / 8 evaluations, plain and with continuous speed 1 + 4g and 1 + 8g, as the reference for the
# flow maps with one duration per position.   bash jobs/eval_teacher_few.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do ck=$DEQ_ROOT/runs/$n/best.pt
  for b in 0 4 8; do o=runs/$n/eval_few_beta$b
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speed base=1 beta=$b power=1 'eval.steps=[2,4,8]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
