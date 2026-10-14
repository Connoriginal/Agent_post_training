# Per-machine settings. Copy to configs/env.sh (gitignored) and edit:
#   cp configs/env.example.sh configs/env.sh
# Every value can also be overridden from the shell, e.g. `NGPUS=2 bash scripts/run_rlvr_grpo.sh`.

# conda env that holds verl + vllm + torch (created by scripts/setup_env.sh)
export CONDA_ENV=${CONDA_ENV:-agent}

# Large artifacts live outside the repo (home is nearly full; /data has room).
export WORK_ROOT=${WORK_ROOT:-/data/$USER/projects/agent_post_training}
export DATA_DIR=${DATA_DIR:-$WORK_ROOT/data}          # preprocessed parquet files
export CKPT_DIR=${CKPT_DIR:-$WORK_ROOT/checkpoints}   # trainer.default_local_dir root
export LOG_DIR=${LOG_DIR:-$WORK_ROOT/logs}            # stdout logs of each run
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}       # model / dataset cache
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$USER}       # keep short: Ray socket paths are length-limited

# Which physical GPUs this run may use. Check `nvidia-smi` first and list only idle ones.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Logging backend for trainer.logger: '["console"]' or '["console","wandb"]' (needs `wandb login`).
export LOGGER=${LOGGER:-'["console"]'}
