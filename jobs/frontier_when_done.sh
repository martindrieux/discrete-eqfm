#!/bin/bash
# Login-node loop: submits the noise-scale sweep (jobs/eval_frontier_maps.sh) for each flow-map run of a task list as
# soon as it is done, four runs per job.   usage: bash jobs/frontier_when_done.sh jobs/tasks_x.tsv
cd ~/discrete-eqfm
names=$(grep -v '^#' $1 | awk -F'\t' '$3!="train_field.py"{print $1}')
sent=""
while true; do
  batch=""; left=0
  for n in $names; do
    case " $sent " in *" $n "*) continue;; esac
    if [ -f runs/$n/DONE ]; then batch="$batch $n"; else left=1; fi
  done
  set -- $batch
  while [ $# -ge 4 ] || { [ $left = 0 ] && [ $# -gt 0 ]; }; do
    grp="${@:1:4}"; sbatch -J front jobs/gpu.sbatch "bash jobs/eval_frontier_maps.sh $grp"; sent="$sent $grp"; shift $(( $# < 4 ? $# : 4 ))
  done
  [ $left = 0 ] && exit 0
  sleep 120
done
