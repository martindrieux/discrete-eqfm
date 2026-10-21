#!/bin/bash
# Extra samplers on the best checkpoint of finished WikiText-2 runs:  bash jobs/eval_extra.sh name1 name2 ...
#   flow maps: jumps with commits (Section 7.2 of the notes) at two thresholds; the commit sampler of DBTM with its
#              repetition exponent (lambda = 1), and with the noise kept
#   denoisers: Euler steps with commits at three thresholds
cd ~/discrete-eqfm && source jobs/env.sh
for n in "$@"; do
  ck=$DEQ_ROOT/runs/$n/best.pt; [ -f $ck ] || ck=$DEQ_ROOT/runs/$n/last.pt
  run() { [ -f runs/$n/$1.json ] || { $PY -u scripts/eval_lm.py $ck out=runs/$n/$1.tmp.json "${@:2}" > runs/$n/$1.log 2>&1 && mv runs/$n/$1.tmp.json runs/$n/$1.json; }; }
  if grep -q '"flow_map": true' runs/$n/eval.json; then
    for h in 0.7 0.9; do run eval_adaptive_h$h sampler=adaptive threshold=$h when=post 'eval.steps=[2,4,8,16]'; done
    run eval_refine_rep1 sampler=refine threshold=0.9 rep=1
    run eval_keepnoise sampler=refine threshold=0.9 fresh=false
  else
    for h in 0.5 0.7 0.9; do run eval_adaptive_h$h sampler=adaptive threshold=$h when=post 'eval.steps=[8,32,128]'; done
  fi
done
