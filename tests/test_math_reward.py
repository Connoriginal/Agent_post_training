import pytest

from agent_post_training.rewards.math_reward import compute_score, final_answer_region


def score(response, gt, **kw):
    return compute_score(data_source="test", solution_str=response, ground_truth=gt, **kw)


@pytest.mark.parametrize(
    "response, gt",
    [
        ("so the answer is \\boxed{\\frac{1}{2}}", "\\frac12"),
        ("\\boxed{0.5}", "\\frac{1}{2}"),
        ("first \\boxed{3} then actually \\boxed{4}", "4"),
        ("reasoning ... </think>\n\nThe answer is \\boxed{10}.", "10"),
    ],
)
def test_correct(response, gt):
    assert score(response, gt)["score"] == 1.0


def test_wrong_answer():
    out = score("\\boxed{5}", "4")
    assert out["score"] == 0.0 and out["answer_found"] == 1.0


def test_no_boxed():
    out = score("the answer is 4", "4")
    assert out["score"] == 0.0 and out["answer_found"] == 0.0


def test_boxed_inside_thinking_does_not_count():
    # reasoning mentions a boxed value, but the final answer after </think> is different
    assert score("maybe \\boxed{4}? no.</think> \\boxed{5}", "4")["score"] == 0.0


def test_truncated_thinking():
    out = score("<think> working \\boxed{4} and then cut", "4")
    assert out["score"] == 0.0 and out["think_closed"] == 0.0
    # opening tag lives in the prompt: only require_think_close can detect the truncation
    assert score("working \\boxed{4} cut", "4")["score"] == 1.0
    assert score("working \\boxed{4} cut", "4", require_think_close=True)["score"] == 0.0


def test_metric_keys_are_stable():
    keys = {frozenset(score(r, "1")) for r in ["\\boxed{1}", "", "<think>x", "</think>1"]}
    assert len(keys) == 1


def test_final_answer_region():
    assert final_answer_region("a</think>b</think>c", False) == "c"
    assert final_answer_region("plain", False) == "plain"
    assert final_answer_region("plain", True) is None
