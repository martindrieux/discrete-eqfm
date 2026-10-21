#!/bin/bash
# Early read of the unfinished TinyStories flow maps under the FMLM+ sampler (files early_*.json, the final ts_*.json are untouched).
cd ~/discrete-eqfm && source jobs/env.sh
names=${*:-TS_eqfm_plain TS_eqfm_context}
for n in $names; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 1.0 0.5; do
    $PY -u scripts/eval_lm.py $CK out=runs/$n/early_rounds_c$c.json 'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.999 noise_scale=$c > runs/$n/early_rounds_c$c.log 2>&1
    $PY -u scripts/eval_lm.py $CK out=runs/$n/early_r4x8_c$c.json 'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 noise_scale=$c > runs/$n/early_r4x8_c$c.log 2>&1
    $PY -u scripts/eval_lm.py $CK out=runs/$n/early_r4x8_churn_c$c.json 'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 churn=0.8 noise_scale=$c > runs/$n/early_r4x8_churn_c$c.log 2>&1
  done
done
