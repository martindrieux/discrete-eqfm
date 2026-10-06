#!/bin/bash
# Submit pool workers.  usage: TASKS=jobs/x.tsv N=4 GPUS=1 HOURS=2 [PART=preempt|earlybird] bash jobs/submit_pool.sh
cd /home/mdrieux/discrete-eqfm
N=${N:-1}; GPUS=${GPUS:-1}; HOURS=${HOURS:-2}; GPT=${GPUS_PER_TASK:-$GPUS}; TASKS=${TASKS:-jobs/tasks.tsv}
if [ "${PART:-preempt}" = earlybird ]; then P="-p flame-earlybirds --qos=earlybird_qos"; else P="-p preempt --qos=preempt_qos"; fi
ARGS="$P --gres=gpu:$GPUS -c $((6 * GPUS)) --mem=$((40 * GPUS))G --time=$HOURS:00:00 -J ${NAME:-deqpool} -o logs/pool-%j.out ${EXTRA}"
for i in $(seq $N); do
  sbatch --parsable $ARGS --export=ALL,TASKS=$TASKS,TLIM_MIN=$((60 * HOURS)),GPUS_PER_TASK=$GPT,RESUBMIT_ARGS="$ARGS" jobs/worker.sh
done
