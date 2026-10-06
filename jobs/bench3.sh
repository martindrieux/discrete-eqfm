T=/tmp/deq_bench_$SLURM_JOB_ID; mkdir -p $T
nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,; nproc; echo "OMP=$OMP_NUM_THREADS"
A="configs/lm1b.yaml train.steps=90 train.log_every=30 train.save_every=1000000 train.warmup=10 train.batch_size=128"
rate() { grep -o '[0-9.]* it/s' $1 | tail -1; }
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/train_field.py $A out_dir=$T/s1 > $T/s1.log 2>&1; echo "BENCH single $(rate $T/s1.log)"
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $PY -u scripts/train_field.py $A out_dir=$T/p$i > $T/p$i.log 2>&1 & done; wait
echo "BENCH four independent: $(for i in 0 1 2 3; do rate $T/p$i.log; done | paste -sd' ')"
for i in 0 1 2 3; do OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$i $PY -u scripts/train_field.py $A out_dir=$T/q$i train.num_workers=1 > $T/q$i.log 2>&1 & done; wait
echo "BENCH four independent omp2 w1: $(for i in 0 1 2 3; do rate $T/q$i.log; done | paste -sd' ')"
L="$PY -m accelerate.commands.launch --multi_gpu --num_processes 4 --mixed_precision no"
$L scripts/train_field.py $A out_dir=$T/d1 > $T/d1.log 2>&1; echo "BENCH ddp4 $(rate $T/d1.log)"
OMP_NUM_THREADS=2 $L scripts/train_field.py $A out_dir=$T/d2 train.num_workers=1 > $T/d2.log 2>&1; echo "BENCH ddp4 omp2 w1 $(rate $T/d2.log)"
L2="$PY -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision no"
OMP_NUM_THREADS=2 $L2 scripts/train_field.py $A out_dir=$T/d3 train.num_workers=1 > $T/d3.log 2>&1; echo "BENCH ddp2 omp2 w1 $(rate $T/d3.log)"
tail -3 $T/d2.log
rm -rf $T
