#!/bin/bash
#SBATCH -p preempt --qos=preempt_qos --gres=gpu:1 -c 16 --mem=64G --time=04:00:00 --requeue
#SBATCH -J deq_setup -o logs/setup-%j.out
# One-off setup on a compute node (the big data disk is not mounted on the login node):
# LM1B blocks, released FLM / FMLM checkpoints, GPT-2 Large and the BERT tokenizer.
cd /home/mdrieux/discrete-eqfm
source jobs/env.sh
mkdir -p $DEQ_ROOT/data $DEQ_ROOT/ckpt/released $DEQ_ROOT/runs $HF_HOME
df -h $DEQ_ROOT | tail -1
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
root = os.environ["DEQ_ROOT"]
for repo in ("FLM-B-LM1B", "FMLM-B-LM1B"):
    p = snapshot_download(f"david3684/{repo}", local_dir=f"{root}/ckpt/released/{repo}")
    print(repo, os.listdir(p), flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
for name in ("gpt2-large", "bert-base-uncased"):
    AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained("gpt2-large")
print("models cached", flush=True)
EOF
[ -f $DEQ_ROOT/data/lm1b_train.bin.json ] || $PY scripts/prepare_lm1b.py --out $DEQ_ROOT/data
ls -la $DEQ_ROOT/data $DEQ_ROOT/ckpt/released/*
echo SETUP_DONE
