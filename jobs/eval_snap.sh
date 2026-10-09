#!/bin/bash
# Evaluate kept checkpoints of a run (deterministic sampler, then churn 0.8 for flow maps).
#   bash jobs/eval_snap.sh NAME STEP      one checkpoint (waits up to an hour for it to appear)
#   bash jobs/eval_snap.sh NAME all SECS  every kept checkpoint not evaluated yet, within SECS seconds
n=$1
one() { # step  [timeout seconds]
  local s=$(printf "%07d" $1) t=${2:-3600}; local ck=$DEQ_ROOT/runs/$n/step_$s.pt o=runs/$n/eval_step$1
  if [ ! -f $o.json ]; then
    timeout ${t}s $PY -u scripts/eval_lm.py $ck out=$o.tmp.json ${EVAL_ARGS} && mv $o.tmp.json $o.json || return 1
  fi
  if grep -q '"flow_map": true' $o.json && [ ! -f ${o}_churn0.8.json ]; then
    timeout ${t}s $PY -u scripts/eval_lm.py $ck out=${o}_churn.tmp.json churn=0.8 && mv ${o}_churn.tmp.json ${o}_churn0.8.json || return 1
  fi
  # the denoiser inside a flow map, sampled alone with Euler steps (clean from-scratch runs)
  if [[ $n == C_* ]] && [ ! -f ${o}_denoiser.json ]; then
    timeout ${t}s $PY -u scripts/eval_lm.py $ck out=${o}_denoiser.tmp.json flow_map=false 'eval.steps=[32,128]' && mv ${o}_denoiser.tmp.json ${o}_denoiser.json || return 1
  fi
  # DBTM sampler (commit and renoise), for the teacher-free runs compared with DBTM
  if [[ $n == T_* || $n == R_* || $n == [PQ]_* ]] && [ ! -f ${o}_refine.json ]; then
    timeout ${t}s $PY -u scripts/eval_lm.py $ck out=${o}_refine.tmp.json sampler=refine threshold=0.9 && mv ${o}_refine.tmp.json ${o}_refine.json || return 1
  fi
  # same sampler, but uncommitted positions keep their own noise instead of a fresh draw
  if [[ $n == [PQ]_* || $n == T_* ]] && [ ! -f ${o}_keepnoise.json ]; then
    timeout ${t}s $PY -u scripts/eval_lm.py $ck out=${o}_keepnoise.tmp.json sampler=refine threshold=0.9 fresh=false && mv ${o}_keepnoise.tmp.json ${o}_keepnoise.json || return 1
  fi
}
if [ "$2" = all ]; then
  end=$(( $(date +%s) + ${3:-600} ))
  for ck in $(ls $DEQ_ROOT/runs/$n/step_*.pt 2>/dev/null); do
    step=$((10#$(basename $ck .pt | cut -d_ -f2))); left=$(( end - $(date +%s) ))
    [ $left -lt 240 ] && break
    one $step $left || break
  done
else
  for i in $(seq 60); do [ -f $DEQ_ROOT/runs/$n/step_$(printf "%07d" $2).pt ] && break; sleep 60; done
  one $2
fi
