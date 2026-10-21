#!/bin/bash
# Sampling tricks on one-step maps, 1 / 2 / 4 / 8 evaluations: commit and renoise (DBTM), partial renoising, commits
# that can be undone, tokens drawn with a temperature, scaled initial noise.   bash jobs/eval_tricks.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/tricks_$tag
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py $CK out=$o.tmp.json 'eval.steps=[1,2,4,8]' "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
for n in "$@"; do
  CK=$DEQ_ROOT/runs/$n/best.pt; [ -f $CK ] || CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 0.5 1.0; do
    run $n commit_c$c            sampler=remask noise_scale=$c
    run $n commit0.999_c$c       sampler=remask noise_scale=$c threshold=0.999
    run $n schedule_c$c          sampler=remask noise_scale=$c threshold=2
    run $n keep0.5_c$c           sampler=remask noise_scale=$c keep=0.5
    run $n keep0.9_c$c           sampler=remask noise_scale=$c keep=0.9
    run $n undo_c$c              sampler=remask noise_scale=$c recommit=true
    run $n undo_keep0.5_c$c      sampler=remask noise_scale=$c recommit=true keep=0.5
    run $n commit_temp0.7_c$c    sampler=remask noise_scale=$c temp=0.7
    run $n keep0.5_temp0.7_c$c   sampler=remask noise_scale=$c keep=0.5 temp=0.7
    run $n jumps_temp0.7_c$c     noise_scale=$c temp=0.7
    run $n jumps_temp1.0_c$c     noise_scale=$c temp=1.0
    run $n churn0.8_c$c          noise_scale=$c churn=0.8
  done
  run $n churn0.8_c1.2 noise_scale=1.2 churn=0.8
done
