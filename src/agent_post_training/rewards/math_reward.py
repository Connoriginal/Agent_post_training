"""Verifiable math reward for RLVR (and for the optional task-reward term in OPD).

Plugged into verl via
    reward.custom_reward_function.path=<this file>
    reward.custom_reward_function.name=compute_score
verl's NaiveRewardManager calls compute_score(data_source=, solution_str=, ground_truth=, extra_info=)
once per rollout, where solution_str is the decoded response (skip_special_tokens=True) and
ground_truth is reward_model.ground_truth from the parquet row. Extra keyword arguments come from
reward.custom_reward_function.reward_kwargs.*.

Score: 1.0 if the last \\boxed{...} in the final-answer region matches the ground truth, else 0.0.
Equivalence = Math-Verify (symbolic) OR verl's string normalization (is_equiv), so either checker
accepting the answer counts as correct.
"""

from verl.utils.reward_score import math_verify
from verl.utils.reward_score.math_reward import is_equiv, last_boxed_only_string, remove_boxed

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def final_answer_region(response: str, require_think_close: bool) -> str | None:
    """Return the part of the response that should contain the final answer, or None if there is none.

    Qwen3.5 thinks inside <think>...</think> by default. The opening tag may sit in the prompt
    (added by the chat template), so the response may contain only the closing tag.
    - "</think>" present          -> text after the last "</think>"
    - "<think>" without a close   -> None (reasoning was cut off by max_response_length)
    - no tags                     -> whole response, unless require_think_close (thinking mode, where a
                                     missing "</think>" also means the reasoning was truncated)
    """
    if THINK_CLOSE in response:
        return response.rsplit(THINK_CLOSE, 1)[1]
    if THINK_OPEN in response or require_think_close:
        return None
    return response


def _string_equiv(boxed: str, ground_truth: str) -> bool:
    try:
        return is_equiv(remove_boxed(boxed), ground_truth)
    except AssertionError:  # remove_boxed asserts on forms like "\fbox{...}"
        return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    require_think_close: bool = False,
    **kwargs,
) -> dict:
    region = final_answer_region(solution_str, require_think_close)
    boxed = last_boxed_only_string(region) if region is not None else None

    correct = False
    if boxed is not None:
        correct = math_verify.compute_score(boxed, ground_truth) > 0 or _string_equiv(boxed, ground_truth)

    score = float(correct)
    # Every key must be present for every sample: verl stacks these into per-batch metric arrays.
    return {
        "score": score,
        "acc": score,
        "answer_found": float(boxed is not None),
        "think_closed": float(region is not None),
    }
