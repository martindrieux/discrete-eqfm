#!/bin/bash
# The sampling tricks of jobs/eval_tricks.sh on the FMLM checkpoint released by its authors (told the time), and the plain
# and continuous-speed samplers on their FLM denoiser.
cd ~/discrete-eqfm && source jobs/env.sh
R=$DEQ_ROOT/ckpt/released
run() { local o=runs/released_fmlm/tricks_$1; shift
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py released:fmlm:$R/FMLM-B-LM1B out=$o.tmp.json 'eval.steps=[1,2,4,8]' "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }; }
run jumps_c1.0
for c in 1.0 0.5 0.7 1.2; do run churn0.8_c$c noise_scale=$c churn=0.8; done
for c in 0.3 0.5 0.7 1.2; do run jumps_c$c noise_scale=$c; done
for c in 1.0 0.5; do run commit_c$c sampler=remask noise_scale=$c; run keep0.5_c$c sampler=remask noise_scale=$c keep=0.5; done
run jumps_temp0.7_c1.0 temp=0.7
for a in "plain" "speed_b16_p2 sampler=speed base=1 beta=16 power=2" "speed_b4_p1 sampler=speed base=1 beta=4 power=1"; do set -- $a; o=runs/released_flm/eval_$1; shift
  [ -f $o.json ] || { $PY -u scripts/eval_lm.py released:flm:$R/FLM-B-LM1B out=$o.tmp.json 'eval.steps=[8,32,128]' "$@" > $o.log 2>&1 && mv $o.tmp.json $o.json; }
done
