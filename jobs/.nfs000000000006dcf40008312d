#!/bin/bash
# Login-node loop: submit an evaluation job for every kept checkpoint that training has passed and that has no result yet.
#   nohup bash jobs/auto_eval.sh "W_ X_ Y_" 12500 > logs/auto_eval.log 2>&1 &
cd /home/mdrieux/discrete-eqfm; mkdir -p logs/.evsub
while true; do
  for d in runs/*; do
    n=$(basename $d); ok=0; for p in $1; do [[ $n == $p* ]] && ok=1; done; [ $ok = 1 ] || continue
    every=$(grep -o '"keep_every": [0-9]*' $d/config.json 2>/dev/null | grep -o '[0-9]*$'); [ -z "$every" ] || [ "$every" = 0 ] && continue
    last=$(grep -a '^step' $d/train.log 2>/dev/null | tail -n 1 | awk '{print $2}'); [ -z "$last" ] && continue
    total=$(grep -o '"steps": [0-9]*' $d/config.json | head -1 | grep -o '[0-9]*$')
    for ((s = every; s <= last && s < total; s += every)); do      # the last step is not a kept checkpoint (last.pt is evaluated by the worker)
      [ -f $d/eval_step$s.json ] && { [ -f $d/eval_step${s}_churn0.8.json ] || ! grep -q '"flow_map": true' $d/eval_step$s.json; } \
        && { [[ $n != T_* && $n != R_* && $n != P_* ]] || [ -f $d/eval_step${s}_refine.json ]; } \
        && { [[ $n != T_* && $n != P_* ]] || [ -f $d/eval_step${s}_keepnoise.json ]; } && continue
      [ -f logs/.evsub/$n.$s ] && [ $(( $(date +%s) - $(stat -c %Y logs/.evsub/$n.$s) )) -lt 7200 ] && continue
      args=""; grep -q '"kind": "field"' $d/config.json 2>/dev/null
      [[ $n == *field* ]] && args="EVAL_ARGS='eval.steps=[32,256,1024]'"
      sbatch --parsable -J aev --time=02:00:00 jobs/gpu.sbatch "$args bash jobs/eval_snap.sh $n $s" > logs/.evsub/$n.$s && echo "$(date +%H:%M) submitted $n $s"
    done
  done
  sleep 600
done
