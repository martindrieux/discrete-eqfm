#!/bin/bash
# Login-node loop: submits jobs/eval_speedmap.sh for each run of a task list as soon as it is done.
cd ~/discrete-eqfm
sent=""
while true; do
  left=0
  for n in $(grep -v '^#' $1 | cut -f1); do
    case " $sent " in *" $n "*) continue;; esac
    if [ -f runs/$n/DONE ]; then sbatch -J spmap jobs/gpu.sbatch "bash jobs/eval_speedmap.sh $n"; sent="$sent $n"; else left=1; fi
  done
  [ $left = 0 ] && exit 0
  sleep 120
done
