#!/bin/bash
# Login-node loop: when a denoiser of tasks_l1 is done, submit the sampler comparison (plain is the worker's own
# evaluation; here: commits and continuous speed laws at 8 / 32 / 128 evaluations).
cd ~/discrete-eqfm
sent=""
while true; do
  left=0
  for n in $(grep -v '^#' jobs/tasks_l1.tsv | cut -f1); do
    case " $sent " in *" $n "*) continue;; esac
    if [ -f runs/$n/DONE ]; then
      sbatch -J law jobs/gpu.sbatch "bash jobs/eval_speed_laws.sh $n '1.0 2 1' '1.0 4 1' '1.0 8 1' '0.75 4 1' '1.0 16 2' '0.75 1024 16' '1.0 1 1'; bash jobs/eval_commit_low.sh $n"
      sent="$sent $n"
    else left=1; fi
  done
  [ $left = 0 ] && exit 0
  sleep 120
done
