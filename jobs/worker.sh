#!/bin/bash
# Pool worker. Takes any task of $TASKS whose dependency is done and which nobody holds, trains it within the job's
# remaining walltime (the trainer checkpoints and exits 3 on its time budget or on preemption), evaluates it when
# training is complete, then takes the next task. Workers resubmit themselves while tasks remain.
#   TASKS (file), TLIM_MIN (job walltime in minutes), GPUS_PER_TASK (default 1), RESUBMIT_ARGS (sbatch options)
# Task line (tab separated):  name <tab> dependency or - <tab> script <tab> arguments
# A task is done when runs/<name>/DONE exists. Checkpoints go to $DEQ_ROOT/runs/<name>, logs and results to runs/<name>.
cd /home/mdrieux/discrete-eqfm
source jobs/env.sh
TASKS=${TASKS:-jobs/tasks.tsv}; GPT=${GPUS_PER_TASK:-1}
END=$(( $(date +%s) + 60 * ${TLIM_MIN:-60} ))
log() { echo "[$(date +%m-%d\ %H:%M:%S) $SLURM_JOB_ID slot$1] ${*:2}"; }
done_() { [ -f runs/$1/DONE ]; }
failed_() { [ -f runs/$1/FAILED ] && [ $(wc -l < runs/$1/FAILED) -ge 3 ]; }
try_lock() {   # atomic mkdir; a lock whose heartbeat is older than 300 s is taken over via an atomic rename
  local d=runs/$1/.lock; mkdir -p runs/$1
  if mkdir $d 2>/dev/null; then echo "$SLURM_JOB_ID" > $d/owner; touch $d/hb; return 0; fi
  local age=$(( $(date +%s) - $(stat -c %Y $d/hb 2>/dev/null || echo 0) ))
  if [ $age -gt 300 ] && mv $d $d.stale.$$ 2>/dev/null; then rm -rf $d.stale.$$; try_lock $1; return $?; fi
  return 1
}

slot() {   # slot <index> <comma-separated gpu ids>
  local id=$1 gpus=$2 idle=0
  while true; do
    local left=$(( (END - $(date +%s)) / 60 - 14 ))      # keep 14 min for the final evaluation and exit
    [ $left -lt 8 ] && { log $id "walltime nearly used"; return 2; }
    local picked="" pending=0
    while IFS=$'\t' read -r name dep script args; do
      [[ -z "$name" || "$name" == \#* ]] && continue
      done_ $name && continue
      failed_ $name && continue
      pending=1
      [ "$dep" != "-" ] && ! done_ $dep && continue
      try_lock $name && { picked=$name; break; }
    done < $TASKS
    [ $pending = 0 ] && { log $id "all tasks done"; return 0; }
    if [ -z "$picked" ]; then
      idle=$((idle + 1)); [ $idle -gt 10 ] && { log $id "idle 10 min, exiting"; return 1; }
      sleep 60; continue
    fi
    idle=0
    script=$(awk -F'\t' -v n=$picked '$1==n{print $3}' $TASKS); args=$(awk -F'\t' -v n=$picked '$1==n{print $4}' $TASKS)
    args=$(eval echo "\"$args\"")
    local launch="$PY -u"
    [ $GPT -gt 1 ] && launch="$PY -m accelerate.commands.launch --multi_gpu --num_processes $GPT --mixed_precision no"
    log $id "start $picked (budget ${left} min, gpus $gpus)"
    ( export CUDA_VISIBLE_DEVICES=$gpus
      $launch scripts/$script $args out_dir=runs/$picked ckpt_dir=$DEQ_ROOT/runs/$picked train.max_minutes=$left
      rc=$?
      if [ $rc = 0 ] && [ ! -f runs/$picked/eval.json ]; then   # bounded by the walltime: an unfinished evaluation is redone
        timeout $(( END - $(date +%s) - 150 ))s $PY -u scripts/eval_lm.py $DEQ_ROOT/runs/$picked/last.pt out=runs/$picked/eval.tmp.json
        rc=$?; [ $rc = 124 ] && rc=3; [ $rc != 0 ] && [ $rc != 3 ] && rc=4
        [ $rc = 0 ] && mv runs/$picked/eval.tmp.json runs/$picked/eval.json
      fi
      exit $rc ) >> runs/$picked/train.log 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do touch runs/$picked/.lock/hb; sleep 30; done
    wait $pid; local rc=$?
    [ $rc = 0 ] && touch runs/$picked/DONE
    rm -rf runs/$picked/.lock
    log $id "end $picked rc=$rc"
    [ $rc = 3 ] && return 2          # time budget hit or preempted: this job is finished
    [ $rc != 0 ] && { echo "rc=$rc $(date) job=$SLURM_JOB_ID" >> runs/$picked/FAILED; sleep 5; }
  done
}

trap 'trap "" TERM; pkill -TERM -P $$ -f scripts/ ; kill -TERM 0; wait' TERM      # preemption: trainers checkpoint and exit
echo "host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION tasks=$TASKS"
resubmit() { ( for v in $(compgen -e | grep '^SLURM_'); do unset $v; done; sbatch --parsable ${RESUBMIT_ARGS} "$@" jobs/worker.sh ); }
if ! $PY -c "import torch; n=torch.cuda.device_count(); assert n > 0; [torch.ones(1, device=f'cuda:{i}').sum().item() for i in range(n)]" 2>/dev/null; then
  log - "GPU unusable on $(hostname); resubmitting elsewhere"; resubmit --exclude=$(hostname); exit 0
fi
NG=$(nvidia-smi -L | wc -l); s=0
for ((g = 0; g + GPT <= NG; g += GPT)); do slot $s $(seq -s, $g $((g + GPT - 1))) & s=$((s + 1)); done
wait
left=0; while IFS=$'\t' read -r name dep script args; do [[ -z "$name" || "$name" == \#* ]] && continue; done_ $name || failed_ $name || left=1; done < $TASKS
[ $left = 1 ] && resubmit > /dev/null && log - "resubmitted replacement"
exit 0
