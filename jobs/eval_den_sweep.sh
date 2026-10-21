#!/bin/bash
# The denoiser teacher under commit-and-renoise: noise scales around 0.5 (where it reaches 9.6 at 32 evaluations),
# and partial renoise. Settings are shared by parallel copies through locks.   bash jobs/eval_den_sweep.sh NAME
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/den_$tag l=runs/$n/den_$tag.lockdir
  [ -f $o.json ] && return
  if ! mkdir $l 2>/dev/null; then
    local own=$(cat $l/owner 2>/dev/null)
    [ "$own" != "$SLURM_JOB_ID" ] && [ -n "$(squeue -h -j $own -t R 2>/dev/null)" ] && return
  fi
  echo $SLURM_JOB_ID > $l/owner
  $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json
  rm -rf $l; }
S='eval.steps=[4,8,16,32,64]'
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 0.3 0.4 0.5 0.6 0.75; do run $n rounds_h0.999_c$c "$S" sampler=refine threshold=0.999 noise_scale=$c; done
  for c in 0.5 0.75; do for k in 0.5 0.75; do
    run $n remask_keep${k}_c$c "$S" sampler=remask threshold=0.999 noise_scale=$c keep=$k; done; done
done
