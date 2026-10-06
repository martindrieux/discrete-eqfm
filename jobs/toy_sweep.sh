#!/bin/bash
# Toy data (8 sequences of 16 tokens): which sampling of jumps and which boundary term make the one-step jump work.
export PATH=$(dirname $PY):$PATH
OUT=runs/toy_sweep; CFG=configs/toy.yaml
python scripts/train_field.py $CFG out_dir=$OUT/field train.steps=4000 > /dev/null
v() { # tag  extra args...
  local tag=$1; shift
  for loss in semigroup lagrangian eulerian; do
    python scripts/train_flow_map.py $CFG out_dir=$OUT/data_${loss}_$tag loss.off_diagonal=$loss train.steps=4000 "$@" > /dev/null &
    python scripts/train_flow_map.py $CFG out_dir=$OUT/distill_${loss}_$tag loss.off_diagonal=$loss loss.teacher=$OUT/field/last.pt train.steps=3000 "$@" > /dev/null &
  done
  wait
}
v notes
v notes_boundary loss.boundary_prob=0.2 loss.boundary_t=0.9
v notes_boundary07 loss.boundary_prob=0.2 loss.boundary_t=0.7
v reparam loss.jump_sampling=reparam loss.diag_time=reparam loss.whole_jump_prob=0.03125
v reparam_boundary loss.jump_sampling=reparam loss.diag_time=reparam loss.whole_jump_prob=0.03125 loss.boundary_prob=0.2 loss.boundary_t=0.9
python scripts/eval_toy.py $OUT 4000
