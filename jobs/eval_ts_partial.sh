#!/bin/bash
# Partial renoise (keep 0.5) on TinyStories maps, noise scales 1 and 0.5.   bash jobs/eval_ts_partial.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/now_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 1.0 0.5; do run $n remask_keep0.5_c$c 'eval.steps=[1,2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=$c keep=0.5; done
  run $n remask_keep0.75_c1.0 'eval.steps=[1,2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=1.0 keep=0.75
done
