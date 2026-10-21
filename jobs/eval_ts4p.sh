#!/bin/bash
# eval_ts4.sh (FMLM+ sampler sweep + partial renoise) split across parallel jobs: each sampler setting is taken by
# the first job that locks it, so K copies of this script on K GPUs share the work.   bash jobs/eval_ts4p.sh NAME
# A lock whose job is no longer running is taken over (a requeued job keeps its id and takes its own lock back).
cd ~/discrete-eqfm && source jobs/env.sh
run() { local n=$1 pre=$2 tag=$3; shift 3; local o=runs/$n/${pre}_$tag l=runs/$n/${pre}_$tag.lockdir
  [ -f $o.json ] && return
  if ! mkdir $l 2>/dev/null; then
    local own=$(cat $l/owner 2>/dev/null)
    [ "$own" != "$SLURM_JOB_ID" ] && [ -n "$(squeue -h -j $own -t R 2>/dev/null)" ] && return
  fi
  echo $SLURM_JOB_ID > $l/owner
  $PY -u scripts/eval_lm.py $CK out=$o.tmp.json "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json
  rm -rf $l; }
for n in "$@"; do CK=$DEQ_ROOT/runs/$n/last.pt
  for c in 1.0 0.5; do run $n now remask_keep0.5_c$c 'eval.steps=[1,2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=$c keep=0.5; done
  run $n now remask_keep0.75_c1.0 'eval.steps=[1,2,4,8,16,32]' sampler=remask threshold=0.999 noise_scale=1.0 keep=0.75
  for c in 1.0 0.5; do
    run $n ts rounds_h0.999_c$c  'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.999 noise_scale=$c
    run $n ts rounds_h0.9_c$c    'eval.steps=[1,2,4,8,16,32]' sampler=refine threshold=0.9 noise_scale=$c
    run $n ts r4x8_h0.999_c$c    'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 noise_scale=$c
    run $n ts r4x8_churn_c$c     'eval.steps=[4]' sampler=refine threshold=0.999 flow_steps=8 churn=0.8 noise_scale=$c
    run $n ts r2x4_h0.999_c$c    'eval.steps=[2,4,8]' sampler=refine threshold=0.999 flow_steps=4 noise_scale=$c
    run $n ts jumps_c$c          'eval.steps=[1,2,4,8,16,32]' noise_scale=$c
    run $n ts churn0.8_c$c       'eval.steps=[2,4,8,16,32]' churn=0.8 noise_scale=$c
  done
done
