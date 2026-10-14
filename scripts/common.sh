# Shared setup sourced by every script in scripts/. Not meant to be executed directly.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# verl is used as an external library checked out at a pinned commit (see setup_env.sh).
VERL_REPO_URL=${VERL_REPO_URL:-https://github.com/verl-project/verl.git}
VERL_COMMIT=${VERL_COMMIT:-76e837e0bfcb18b380a66c0ba56ef9f0ac840f09}  # main @ 2026-10-13, version 0.10.0.dev
VERL_DIR=${VERL_DIR:-$REPO_ROOT/third_party/verl}

if [ -f "$REPO_ROOT/configs/env.sh" ]; then
    source "$REPO_ROOT/configs/env.sh"
else
    source "$REPO_ROOT/configs/env.example.sh"
fi

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
}

# Number of GPUs listed in CUDA_VISIBLE_DEVICES.
count_visible_gpus() {
    awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
}
