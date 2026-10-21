#!/bin/bash
# Doubled-token rate of samples (real text, plain steps, commits, soft speed laws) on one WikiText-2 denoiser, 32 steps.
cd ~/discrete-eqfm && source jobs/env.sh
n=$1; ck=$DEQ_ROOT/runs/$n/best.pt; S="eval.steps=[32]"
[ -f runs/baselines/wt2_test_dup.json ] || $PY -u scripts/eval_lm.py data:data/wt2_test.bin out=runs/baselines/wt2_test_dup.json > /dev/null 2>&1
ev() { o=runs/$n/dup_$1; shift; [ -f $o.json ] || { $PY -u scripts/eval_lm.py $ck out=$o.tmp.json "$S" "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
ev plain
for h in 0.3 0.4 0.5 0.7; do ev commit_h$h sampler=adaptive threshold=$h; done
for law in "0.75 4 1" "0.75 2 1" "0.75 16 2" "0.75 64 4" "0.75 1024 16" "1.0 2 1" "0.75 1 1"; do set -- $law; ev speed_b$1_beta$2_p$3 sampler=speed base=$1 beta=$2 power=$3; done
