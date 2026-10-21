#!/bin/bash
# Ablation of the adaptive-speed sampler: the network is told one shared duration per sequence (shared=true) while each
# position still moves by its own amount; compare with eval_speedmap_beta*.json.   bash jobs/eval_speedmap_shared.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for b in 2 8; do o=runs/$n/eval_speedmap_shared_beta$b
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json sampler=speedmap beta=$b shared=true 'eval.steps=[8,16]' > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
