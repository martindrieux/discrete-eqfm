#!/bin/bash
# Login-node loop: the TinyStories workers may start before the data is prepared and mark the first task as failed.
# Once the data evaluation exists, the failure mark is cleared and a worker is submitted if none is queued.
cd ~/discrete-eqfm
until [ -f data/tinystories_train.bin.json ]; do sleep 60; done
if [ -f runs/TS_denoiser_context/FAILED ]; then
  mv runs/TS_denoiser_context/FAILED runs/TS_denoiser_context/FAILED.before_data
fi
if ! squeue -u $USER -h -n ts | grep -q .; then
  TASKS=jobs/tasks_ts.tsv N=2 GPUS=4 HOURS=8 NAME=ts bash jobs/submit_pool.sh
fi
