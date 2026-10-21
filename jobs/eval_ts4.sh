#!/bin/bash
# TinyStories final evaluation: the FMLM+ sampler sweep (eval_ts.sh) and partial renoise (eval_ts_partial.sh).   bash jobs/eval_ts4.sh NAME
bash ~/discrete-eqfm/jobs/eval_ts.sh "$@"
bash ~/discrete-eqfm/jobs/eval_ts_partial.sh "$@"
