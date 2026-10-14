# Agent_post_training

RLVR (GRPO) and on-policy distillation (OPD) of **Qwen3.5-4B** on math, built on [verl](https://github.com/verl-project/verl).

- Train: MATH (`DigitalLearningGmbH/MATH-lighteval`, 7,498 problems) · Val: MATH-500
- Reward: last `\boxed{}` answer after `</think>`, checked with Math-Verify / verl normalization (1 or 0)
- OPD teacher: **not chosen yet** — set `TEACHER_MODEL` (must be a Qwen3.5-family model, same tokenizer)

## Quick start

```bash
bash scripts/setup_env.sh                  # conda env `agent` + verl (pinned commit) + vLLM + FSDP
cp configs/env.example.sh configs/env.sh   # edit paths / CUDA_VISIBLE_DEVICES / logger
bash scripts/prepare_data.sh
python -m pytest tests

bash scripts/run_rlvr_grpo.sh                          # RLVR (GRPO)
TEACHER_MODEL=<teacher> bash scripts/run_opd.sh        # OPD
DRY_RUN=1 bash scripts/run_rlvr_grpo.sh                # show resolved config only
```

A Korean write-up of how verl works, every setting used here, and the verification results is kept locally in
`report/verl_guide.pdf` (gitignored; rebuild with `bash report/build.sh`).
