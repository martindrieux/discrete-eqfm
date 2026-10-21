#!/bin/bash
# TinyStories flow maps under the sampler of FMLM+ and its EqFM variants.   bash jobs/eval_ts.sh NAME...
#   headline of FMLM+: 4 rounds x 8 flow steps, commit above 0.999, fresh noise (32 evaluations: 17.53 / 3.96)
#   their sweep: 1 to 32 evaluations (96.28/3.97, 35.14/3.98, 17.76/3.91, 11.57/3.86, 8.94/3.85, 7.40/3.83)
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/${PREFIX:-ts}_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 1.0 0.5; do
    run $n rounds_h0.999_c$c  'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.999 noise_scale=$c
    run $n rounds_h0.9_c$c    'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.9 noise_scale=$c
    run $n r4x8_h0.999_c$c    'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 noise_scale=$c
    run $n r4x8_churn_c$c     'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 churn=0.8 noise_scale=$c
    run $n r2x4_h0.999_c$c    'eval.steps=[2,4,8]' sampler=refine threshold=0.999 flow_steps=4 noise_scale=$c
    run $n jumps_c$c          'eval.steps=[1,2,4,8,16,32]' noise_scale=$c
    run $n churn0.8_c$c       'eval.steps=[2,4,8,16,32]' churn=0.8 noise_scale=$c
  done
done
