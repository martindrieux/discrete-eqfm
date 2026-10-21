#!/bin/bash
# FMLM stochastic sampler (churn gamma) with a scaled initial noise, 2 / 4 / 8 evaluations.   bash jobs/eval_churn_noise.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  for g in 0.8 0.5 0.9 1.0; do for c in 0.3 0.4 0.6 0.7; do o=runs/$n/tricks_churn${g}_c$c
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json 'eval.steps=[2,4,8]' churn=$g noise_scale=$c > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done; done
  for g in 0.5 0.9 1.0; do o=runs/$n/tricks_churn${g}_c0.5
    [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json 'eval.steps=[2,4,8]' churn=$g noise_scale=0.5 > $o.log 2>&1 && mv $o.tmp.json $o.json; }
  done
done
