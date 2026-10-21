#!/bin/bash
# Partial renoise: larger share kept, on TinyStories maps.   bash jobs/eval_ts_keep.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/now_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  run $n remask_keep0.9_c1.0   'eval.steps=[2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=1.0 keep=0.9
  run $n remask_keep0.75_c0.75 'eval.steps=[2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=0.75 keep=0.75
done
