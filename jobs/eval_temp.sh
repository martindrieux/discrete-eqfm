#!/bin/bash
# Partial renoise with drawn commits (temp > 0 draws from p^(1/temp)): flow maps at matched entropy.   bash jobs/eval_temp.sh NAME...
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 tag=$2; shift 2; local o=runs/$n/tmp_$tag l=runs/$n/tmp_$tag.lockdir
  [ -f $o.json ] && return
  if ! mkdir $l 2>/dev/null; then
    local own=$(cat $l/owner 2>/dev/null)
    [ "$own" != "$SLURM_JOB_ID" ] && [ -n "$(squeue -h -j $own -t R 2>/dev/null)" ] && return
  fi
  echo $SLURM_JOB_ID > $l/owner
  $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json
  rm -rf $l; }
S='eval.steps=[4,8,16,32]'
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for T in 1.0 0.8 0.6; do
    run $n remask_keep0.75_c1.0_T$T "$S" sampler=remask threshold=0.999 noise_scale=1.0 keep=0.75 temp=$T
    run $n remask_keep0.5_c0.5_T$T  "$S" sampler=remask threshold=0.999 noise_scale=0.5 keep=0.5 temp=$T
  done
done
