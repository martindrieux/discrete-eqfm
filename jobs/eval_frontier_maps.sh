#!/bin/bash
# Perplexity against entropy for one-step maps on WikiText-2, by scaling the initial noise (the entropy knob of the
# DBTM paper): Beckmann plain and with refinement in the loop, FMLM, and EqFM with no auxiliary loss. 1 and 8 evaluations.
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for s in 0.7 0.8 0.9 1.0 1.1 1.2 1.35 1.5; do
    o=runs/$n/frontier_s$s
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json 'eval.steps=[1,8]' noise_scale=$s > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
