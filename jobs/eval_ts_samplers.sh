#!/bin/bash
# Sampler variants that helped on LM1B, on TinyStories maps.   bash jobs/eval_ts_samplers.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/now_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
S='eval.steps=[1,2,4,8,16,32]'
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  run $n remask_keep0.5_c0.5   "$S" sampler=remask threshold=0.999 noise_scale=0.5 keep=0.5
  run $n remask_keep0.5_c1.0   "$S" sampler=remask threshold=0.999 noise_scale=1.0 keep=0.5
  run $n rounds_h0.99_c0.5     "$S" sampler=refine threshold=0.99 noise_scale=0.5
  run $n rounds_h0.9999_c0.5   "$S" sampler=refine threshold=0.9999 noise_scale=0.5
  run $n speedmap_b2           'eval.steps=[8,16,32]' sampler=speedmap beta=2
  run $n speedmap_b8           'eval.steps=[8,16,32]' sampler=speedmap beta=8
done
