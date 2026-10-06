# Sourced by every job. Big files live on /data/user_data (compute nodes only); the repo and small logs live in home.
export DEQ_ROOT=/data/user_data/mdrieux/deqfm
export HF_HOME=$DEQ_ROOT/hf
export PY=/home/mdrieux/discrete-eqfm/.venv/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
