#!/bin/bash
# Throughput and peak memory of each objective on one GPU (LM1B, 12-layer DiT).
T=/tmp/deq_bench_$SLURM_JOB_ID; mkdir -p $T
run() { # name script args...
  local name=$1 script=$2; shift 2
  $PY - "$name" "$script" "$@" <<'PYEOF'
import subprocess, sys, re, os, time
name, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
T = os.environ.get("T")
cmd = [sys.executable, f"scripts/{script}", "configs/lm1b.yaml", f"out_dir={T}/{name}", "train.steps=80", "train.log_every=20",
       "train.save_every=1000000", "train.num_workers=2", "train.warmup=10", *args]
env = dict(os.environ, DEQ_REPORT_MEM="1")
p = subprocess.run(cmd, capture_output=True, text=True, env=env)
rates = re.findall(r"([\d.]+) it/s", p.stdout)
mem = re.findall(r"peak memory ([\d.]+) GB", p.stdout)
print(f"BENCH {name:40s} it/s {rates[-1] if rates else 'FAIL'} mem {mem[-1] if mem else '?'} GB", flush=True)
if not rates: print(p.stdout[-1500:], p.stderr[-3000:], flush=True)
PYEOF
}
export T
run field_b128 train_field.py train.batch_size=128
run field_b256 train_field.py train.batch_size=256
run field_b128_nodrop train_field.py train.batch_size=128 model.dropout=0
$PY scripts/train_field.py configs/lm1b.yaml out_dir=$T/teacher train.steps=20 train.save_every=20 train.batch_size=64 train.warmup=5 > /dev/null 2>&1
for off in semigroup lagrangian eulerian; do
  run data_${off}_b128 train_flow_map.py train.batch_size=128 loss.off_diagonal=$off
  run distill_${off}_b128 train_flow_map.py train.batch_size=128 loss.off_diagonal=$off loss.teacher=$T/teacher/last.pt
done
run data_semigroup_b256 train_flow_map.py train.batch_size=256
run data_lagrangian_fd_b128 train_flow_map.py train.batch_size=128 loss.off_diagonal=lagrangian loss.jvp=fd
run data_eulerian_fd_b128 train_flow_map.py train.batch_size=128 loss.off_diagonal=eulerian loss.jvp=fd
run time_semigroup_b128 train_flow_map.py train.batch_size=128 model.cond=time
run statetime_semigroup_b128 train_flow_map.py train.batch_size=128 model.cond=state_time
rm -rf $T
