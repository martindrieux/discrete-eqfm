#!/bin/bash
# Posterior refinement as in FMLM+ (commit above 0.999, else top-k on a linear schedule, fresh noise) at 8 to 128 rounds.
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for c in 1.0 0.5; do for h in 0.999 0.9; do o=runs/$n/posterior_h${h}_c$c
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json 'eval.steps=[8,16,32,64,128]' sampler=remask threshold=$h noise_scale=$c > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done; done
done
