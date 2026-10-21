#!/bin/bash
# Login-node loop: for each run of a task list, submits K copies of `bash $2 <run>` (1 GPU, 2 h each) as soon as the
# run is done; the copies split the work through the locks of $2.   K=4 bash jobs/chain_par.sh TASKS SCRIPT
cd ~/discrete-eqfm; sent=""; K=${K:-4}
while true; do left=0
  for n in $(grep -v '^#' $1 | cut -f1); do
    case " $sent " in *" $n "*) continue;; esac
    if [ -f runs/$n/DONE ]; then
      for i in $(seq $K); do sbatch -J chain --time=02:00:00 jobs/gpu.sbatch "bash $2 $n"; done; sent="$sent $n"
    else left=1; fi
  done
  [ $left = 0 ] && exit 0; sleep 120
done
