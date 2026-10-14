#!/usr/bin/env bash
# Download MATH (train) + MATH-500 (val) and write verl parquet files to $DATA_DIR/math.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

python -m agent_post_training.data.prepare_math --out_dir "$DATA_DIR/math" "$@"
