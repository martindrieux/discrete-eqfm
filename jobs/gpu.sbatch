#!/bin/bash
#SBATCH -p preempt --qos=preempt_qos --gres=gpu:1 -c 6 --mem=48G --time=04:00:00 --requeue
#SBATCH -J deq -o logs/%x-%j.out
# Runs the given command line on one GPU:  sbatch -J name jobs/gpu.sbatch <command ...>
cd /home/mdrieux/discrete-eqfm
source jobs/env.sh
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) job=$SLURM_JOB_ID"
eval "$@"
echo "EXIT=$?"
