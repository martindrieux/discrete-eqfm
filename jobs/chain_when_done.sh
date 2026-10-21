#!/bin/bash
# Login-node loop: for each run of a task list, submits `bash $2 <run>` on one GPU as soon as the run is done.
cd ~/discrete-eqfm; sent=""
while true; do left=0
  for n in $(grep -v '^#' $1 | cut -f1); do
    case " $sent " in *" $n "*) continue;; esac
    if [ -f runs/$n/DONE ]; then sbatch -J chain jobs/gpu.sbatch "bash $2 $n"; sent="$sent $n"; else left=1; fi
  done
  [ $left = 0 ] && exit 0; sleep 120
done
