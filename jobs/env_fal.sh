# fal-h100 (shared 8x H100): big files on /shared (EFS), runs/ and data/ are links to it.
export DEQ_ROOT=/shared/martin/deqfm
export HF_HOME=$DEQ_ROOT/hf
export PY=$DEQ_ROOT/venv/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
