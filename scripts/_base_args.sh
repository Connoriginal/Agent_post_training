# Hydra overrides shared by run_rlvr_grpo.sh and run_opd.sh. Sourced after common.sh; every value
# is an env var with a default so a run can be tuned from the shell without editing files.
#
# Defaults target Qwen3.5-4B (dense, 32 layers, Gated DeltaNet + gated attention hybrid) on
# H100 94GB GPUs. Every key below is a field of verl's ppo_trainer.yaml; see report/ for what each does.

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/math/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/math/test.parquet}
REWARD_FN=${REWARD_FN:-$REPO_ROOT/src/agent_post_training/rewards/math_reward.py}

# Qwen3.5 thinks by default. Thinking traces need long responses (>=8k); off by default to keep
# rollouts short. Turning it on also makes the reward demand a closed </think>.
ENABLE_THINKING=${ENABLE_THINKING:-False}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}         # prompts per step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}    # prompts per optimizer step -> 2 updates per step
ACTOR_LR=${ACTOR_LR:-1e-6}
# token budget per GPU for dynamic batching; must be >= prompt + response length
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( 4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) ))}

# FSDP param/optimizer CPU offload. Needed when the actor is not sharded across enough GPUs
# (e.g. a single GPU: fp32 params + Adam state + vLLM do not fit together in 94GB).
OFFLOAD=${OFFLOAD:-False}

ROLLOUT_TP=${ROLLOUT_TP:-1}                       # 4B fits on one GPU
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}

PROJECT_NAME=${PROJECT_NAME:-agent_post_training}

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FN"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.require_think_close=$ENABLE_THINKING
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.max_model_len=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
)

# Fills in TRAINER once EXPERIMENT_NAME and the student GPU count are known.
build_trainer_args() {
    local experiment_name=$1 n_gpus=$2
    TRAINER=(
        trainer.logger="$LOGGER"
        trainer.project_name="$PROJECT_NAME"
        trainer.experiment_name="$experiment_name"
        trainer.default_local_dir="$CKPT_DIR/$PROJECT_NAME/$experiment_name"
        trainer.validation_data_dir="$LOG_DIR/$experiment_name/val_generations"
        trainer.n_gpus_per_node=$n_gpus
        trainer.nnodes=1
        trainer.critic_warmup=0
        trainer.balance_batch=True
        trainer.val_before_train=True
        trainer.resume_mode=auto
        trainer.save_freq=$SAVE_FREQ
        trainer.test_freq=$TEST_FREQ
        trainer.total_epochs=$TOTAL_EPOCHS
    )
}

# Run verl's PPO entry point with the conda interpreter (driver and Ray workers), tee'ing to LOG_DIR.
# DRY_RUN=1 prints the resolved Hydra config instead of training.
launch_main_ppo() {
    local experiment_name=$1; shift
    mkdir -p "$LOG_DIR/$experiment_name" "$RAY_TMPDIR"
    local log_file="$LOG_DIR/$experiment_name/train_$(date +%Y%m%d_%H%M%S).log"
    cd "$VERL_DIR"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        python3 -m verl.trainer.main_ppo "$@" --cfg job --resolve
        return
    fi
    echo "log: $log_file"
    python3 -m verl.trainer.main_ppo "$@" 2>&1 | tee "$log_file"
}
