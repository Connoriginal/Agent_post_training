#!/usr/bin/env bash
# RLVR with GRPO: Qwen3.5-4B on MATH, binary verifiable reward from rewards/math_reward.py.
#
#   bash scripts/run_rlvr_grpo.sh                         # all GPUs in CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_rlvr_grpo.sh  # single GPU
#   DRY_RUN=1 bash scripts/run_rlvr_grpo.sh               # print resolved config only
#   bash scripts/run_rlvr_grpo.sh trainer.total_epochs=1  # extra Hydra overrides are appended last
#
# Long run: start inside tmux so it survives disconnects.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$REPO_ROOT/scripts/_base_args.sh"
activate_env

NGPUS=${NGPUS:-$(count_visible_gpus)}
ROLLOUT_N=${ROLLOUT_N:-8}               # GRPO group size: responses per prompt
USE_KL_LOSS=${USE_KL_LOSS:-True}        # KL to the frozen initial policy, added to the actor loss
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_4b_grpo_math_think${ENABLE_THINKING}}

ALGO=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
)

build_trainer_args "$EXPERIMENT_NAME" "$NGPUS"

launch_main_ppo "$EXPERIMENT_NAME" \
    "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
    "${ALGO[@]}" "${TRAINER[@]}" "$@"
