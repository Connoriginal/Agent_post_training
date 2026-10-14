"""Build verl RL parquet files for math RLVR / OPD.

train: DigitalLearningGmbH/MATH-lighteval  train split (7,500 problems, MATH competition)
val:   HuggingFaceH4/MATH-500              test split  (500 problems, subset of the MATH test split,
                                                          so it does not overlap the train split)

Each row follows verl's RLHFDataset schema:
    data_source  str   -> used for per-source validation metrics and (multi-teacher) OPD routing
    prompt       list  -> chat messages; verl applies the model's chat template
    ability      str
    reward_model dict  -> {"style": "rule", "ground_truth": <final answer string>}
    extra_info   dict  -> free-form, passed to the reward function

Usage:
    python -m agent_post_training.data.prepare_math --out_dir $DATA_DIR/math
"""

import argparse
import json
import os

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

TRAIN_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
VAL_SOURCE = "HuggingFaceH4/MATH-500"


def to_row(question: str, answer: str, data_source: str, split: str, idx: int, extra: dict) -> dict:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": split, "index": idx, **extra},
    }


def extract_boxed(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed) if boxed else ""


def build_train() -> datasets.Dataset:
    ds = datasets.load_dataset(TRAIN_SOURCE, split="train")

    def fn(ex, idx):
        answer = extract_boxed(ex["solution"])
        return to_row(ex["problem"], answer, TRAIN_SOURCE, "train", idx, {"level": ex["level"], "type": ex["type"]})

    return ds.map(fn, with_indices=True, remove_columns=ds.column_names)


def build_val() -> datasets.Dataset:
    ds = datasets.load_dataset(VAL_SOURCE, split="test")

    def fn(ex, idx):
        extra = {"level": str(ex["level"]), "type": ex["subject"], "unique_id": ex["unique_id"]}
        return to_row(ex["problem"], ex["answer"], VAL_SOURCE, "test", idx, extra)

    return ds.map(fn, with_indices=True, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for name, ds in (("train", build_train()), ("test", build_val())):
        n_before = len(ds)
        # A reference solution without \boxed{} gives an empty answer: unusable for a verifiable reward.
        ds = ds.filter(lambda ex: bool(ex["reward_model"]["ground_truth"]))
        path = os.path.join(out_dir, f"{name}.parquet")
        ds.to_parquet(path)
        print(f"{name}: kept {len(ds)}/{n_before} rows -> {path}")
        with open(os.path.join(out_dir, f"{name}_example.json"), "w") as f:
            json.dump(ds[0], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
