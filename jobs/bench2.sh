T=/tmp/deq_bench_$SLURM_JOB_ID; mkdir -p $T
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
b() { name=$1; np=$2; shift 2
  if [ $np = 1 ]; then L="$PY -u"; else L="$PY -m accelerate.commands.launch --multi_gpu --num_processes $np --mixed_precision no"; fi
  DEQ_REPORT_MEM=1 $L scripts/train_field.py configs/lm1b.yaml out_dir=$T/$name train.steps=120 train.log_every=40 train.save_every=1000000 train.warmup=10 "$@" > $T/$name.log 2>&1
  echo "BENCH $name $(grep -o '[0-9.]* it/s' $T/$name.log | tail -1) $(grep -o 'peak memory [0-9.]* GB' $T/$name.log | tail -1) $(grep -o 'OutOfMemory' $T/$name.log | head -1)"; }
b g1_b128 1 train.batch_size=128
b g1_b256 1 train.batch_size=256
b g1_b512 1 train.batch_size=512
b g2_b128 2 train.batch_size=128
b g2_b256 2 train.batch_size=256
b g2_b128_w0 2 train.batch_size=128 train.num_workers=0
rm -rf $T
