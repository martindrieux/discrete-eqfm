#!/bin/bash
# Evaluate a kept checkpoint of a run:  bash jobs/eval_snap.sh NAME STEP   (deterministic sampler, then churn 0.8 for flow maps)
n=$1; s=$(printf "%07d" $2); ck=$DEQ_ROOT/runs/$n/step_$s.pt
for i in $(seq 60); do [ -f $ck ] && break; sleep 60; done
[ -f $ck ] || { echo "no $ck"; exit 1; }
[ -f runs/$n/eval_step$2.json ] || $PY -u scripts/eval_lm.py $ck out=runs/$n/eval_step$2.json
grep -q '"flow_map": true' runs/$n/eval_step$2.json && { [ -f runs/$n/eval_step$2_churn0.8.json ] || $PY -u scripts/eval_lm.py $ck out=runs/$n/eval_step$2_churn0.8.json churn=0.8; }
