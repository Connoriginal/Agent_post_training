#!/usr/bin/env bash
# On-policy distillation (OPD): Qwen3.5-4B student samples on MATH prompts, a frozen teacher scores
# every student token, and the student is trained toward the teacher (verl `distillation.*`).
#
# The teacher is intentionally left blank: set TEACHER_MODEL before running. It must share the
# student's tokenizer/vocab (i.e. another Qwen3.5 model).
#
#   TEACHER_MODEL=<hf id or path> bash scripts/run_opd.sh
#   TEACHER_MODEL=... STUDENT_GPUS=2 TEACHER_GPUS=2 bash scripts/run_opd.sh
#   TEACHER_MODEL=... DRY_RUN=1 bash scripts/run_opd.sh     # print resolved config only
#
# GPUs: the first STUDENT_GPUS visible GPUs train/roll out the student, the remaining TEACHER_GPUS
# serve the teacher (separate Ray resource pool). STUDENT_GPUS + TEACHER_GPUS must equal the
# number of GPUs in CUDA_VISIBLE_DEVICES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$REPO_ROOT/scripts/_base_args.sh"
activate_env

TEACHER_MODEL=${TEACHER_MODEL:-}   # TODO: choose the teacher (e.g. a larger Qwen3.5 model)
if [ -z "$TEACHER_MODEL" ]; then
    echo "TEACHER_MODEL is empty. Set it to a Qwen3.5-family model (same tokenizer as $MODEL_PATH)." >&2
    exit 1
fi

N_VISIBLE=$(count_visible_gpus)
TEACHER_GPUS=${TEACHER_GPUS:-$(( N_VISIBLE / 2 ))}
STUDENT_GPUS=${STUDENT_GPUS:-$(( N_VISIBLE - TEACHER_GPUS ))}
if [ $(( STUDENT_GPUS + TEACHER_GPUS )) -ne "$N_VISIBLE" ] || [ "$TEACHER_GPUS" -lt 1 ]; then
    echo "STUDENT_GPUS($STUDENT_GPUS) + TEACHER_GPUS($TEACHER_GPUS) must equal visible GPUs ($N_VISIBLE), teacher >= 1." >&2
    exit 1
fi
TEACHER_TP=${TEACHER_TP:-1}                         # replicas = TEACHER_GPUS / TEACHER_TP (auto-filled by verl)
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.8}   # teacher GPUs are dedicated

# PG OPD (default): reward_t = log p_teacher(y_t) - log p_student(y_t) (k1), PPO-clipped policy gradient.
# GKD OPD: DISTILL_LOSS_MODE=forward_kl_topk USE_POLICY_GRADIENT=False (teacher top-k forward KL).
DISTILL_LOSS_MODE=${DISTILL_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILL_TOPK=${DISTILL_TOPK:-64}
# False: pure distillation. True: add the GRPO task-reward loss (then raise ROLLOUT_N above 1).
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
DISTILL_LOSS_COEF=${DISTILL_LOSS_COEF:-1.0}
ROLLOUT_N=${ROLLOUT_N:-1}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_4b_opd_${DISTILL_LOSS_MODE}_math_think${ENABLE_THINKING}}

ALGO=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.actor.use_kl_loss=False       # don't also pull toward the initial policy
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.rollout.calculate_log_probs=True
)

DISTILL=(
    distillation.enabled=True
    distillation.nnodes=1
    distillation.n_gpus_per_node=$TEACHER_GPUS
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEM_UTIL
    distillation.teacher_models.teacher_model.inference.max_model_len=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
    distillation.distillation_loss.loss_mode=$DISTILL_LOSS_MODE
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.topk=$DISTILL_TOPK
    distillation.distillation_loss.use_task_rewards=$USE_TASK_REWARDS
    distillation.distillation_loss.distillation_loss_coef=$DISTILL_LOSS_COEF
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

build_trainer_args "$EXPERIMENT_NAME" "$STUDENT_GPUS"

launch_main_ppo "$EXPERIMENT_NAME" \
    "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
    "${ALGO[@]}" "${DISTILL[@]}" "${TRAINER[@]}" "$@"
