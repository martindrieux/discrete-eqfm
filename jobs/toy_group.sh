#!/bin/bash
# Trains and evaluates several toy runs side by side on one GPU.
#   bash jobs/toy_group.sh CONFIG SCRIPT "name1|overrides1" "name2|overrides2" ...      (inside a GPU job)
# Results: runs/<name>/toy_eval.{txt,json}. Checkpoints stay on the data disk.
cfg=$1; script=$2; shift 2
for spec in "$@"; do
  name=${spec%%|*}; ov=${spec#*|}
  ( [ -f runs/$name/toy_eval.json ] && exit 0
    mkdir -p runs/$name
    $PY -u scripts/$script $cfg $ov out_dir=runs/$name ckpt_dir=$DEQ_ROOT/runs/$name > runs/$name/train.log 2>&1 &&
    $PY -u scripts/eval_toy_markov.py $DEQ_ROOT/runs/$name/last.pt out=runs/$name/toy_eval.json ${EVAL_OPTS} > runs/$name/toy_eval.txt 2>&1
    echo "$name rc=$?" ) &
done
wait
