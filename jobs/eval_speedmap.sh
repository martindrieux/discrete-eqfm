#!/bin/bash
# Adaptive-speed sampler of the flow map with one duration per position (sampling.sample_speed_map), speed 1 + beta g,
# at 4 / 8 / 16 evaluations; beta = 0 is the plain sampler.   bash jobs/eval_speedmap.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for b in 0 1 2 4 8 16; do o=runs/$n/eval_speedmap_beta$b
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speedmap beta=$b 'eval.steps=[4,8,16]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
