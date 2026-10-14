#!/usr/bin/env bash
# Create the `agent` conda env and install verl (pinned commit) with the vllm + fsdp + math extras.
#
# verl's own workflow is `uv run` from its repo root (a .venv built from uv.lock). We keep a conda env
# instead, so the exact package set is exported from verl's committed uv.lock and installed into conda
# with `uv pip`; verl itself is installed editable from third_party/verl. Launch scripts then run with
# VERL_USE_UV=0 semantics (plain python3, Ray workers inherit the conda interpreter).
#
# Re-running is safe: existing env / checkout are reused.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" python=3.12   # verl's uv.lock targets CPython 3.12
fi
conda activate "$CONDA_ENV"
python -m pip install -q -U uv

if [ ! -d "$VERL_DIR/.git" ]; then
    git clone "$VERL_REPO_URL" "$VERL_DIR"
fi
git -C "$VERL_DIR" fetch -q origin "$VERL_COMMIT" 2>/dev/null || true
git -C "$VERL_DIR" checkout -q --detach "$VERL_COMMIT"

LOCK_TXT="$VERL_DIR/.conda-requirements.txt"
(cd "$VERL_DIR" && uv export --quiet --frozen --all-packages --no-hashes --no-emit-workspace \
    --extra vllm --extra fsdp --extra math -o "$LOCK_TXT")

# The export is already the full resolved closure, so install it as-is (--no-deps): re-resolving would
# ignore verl's [tool.uv].override-dependencies (e.g. its numpy pin) and fail.
# uv export drops index routing, so re-add the two explicit indexes from verl's pyproject.toml:
# torch +cu130 wheels and the verl wheelhouse (prebuilt flash-attn etc.).
uv pip install --python "$(command -v python)" --no-deps -r "$LOCK_TXT" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple/ \
    --index-strategy unsafe-best-match
uv pip install --python "$(command -v python)" --no-deps -e "$VERL_DIR"
# this repo (rewards, data prep) as an importable package
uv pip install --python "$(command -v python)" --no-deps -e "$REPO_ROOT"

python - <<'PY'
import torch, vllm, transformers, verl, math_verify  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__, "| verl", verl.__version__)
PY
