#!/usr/bin/env bash
# End-to-end check on the toy data (CPU): the field, then each off-diagonal loss, distilled and from data.
#   bash scripts/toy_validation.sh [out_dir]
set -e
OUT=${1:-runs/toy_validation}
CFG=configs/toy.yaml
python scripts/train_field.py $CFG out_dir=$OUT/field train.steps=4000
for loss in semigroup lagrangian eulerian; do
  python scripts/train_flow_map.py $CFG out_dir=$OUT/distill_$loss loss.off_diagonal=$loss \
         loss.teacher=$OUT/field/last.pt train.steps=3000
  python scripts/train_flow_map.py $CFG out_dir=$OUT/data_$loss loss.off_diagonal=$loss train.steps=4000
done
python scripts/eval_toy.py $OUT
