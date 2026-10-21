#!/bin/bash
# Noise-scale sweep of the commit-and-renoise sampler on TinyStories maps.   bash jobs/eval_ts_noise.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/now_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 0.25 0.0; do
    run $n rounds_h0.999_c$c 'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.999 noise_scale=$c
  done
done
