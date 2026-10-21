#!/bin/bash
# Denoisers at 4 / 8 / 32 evaluations with continuous speed 1 + beta g (beta = 0: plain sampler).   bash jobs/eval_den_speed.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for b in 0 4 8; do o=runs/$n/eval_den_beta$b
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speed base=1 beta=$b power=1 'eval.steps=[4,8,32]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
