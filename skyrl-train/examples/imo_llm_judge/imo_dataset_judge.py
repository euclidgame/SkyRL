# Converts Tinker IMO AnswerBench JSONL into SkyRL parquet format for LLM-as-a-judge training.
#
# Example:
#   uv run examples/imo_llm_judge/imo_dataset_judge.py \
#     --train_jsonl /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32-trainset.jsonl \
#     --val_jsonl   /accounts/projects/berkeleynlp/windsey/projects/tinker-cookbook/training_data_imo/answerbench-hard32.jsonl \
#     --output_dir  $HOME/data/imo_llm_judge

import argparse
import os

import datasets


IMO_SUFFIX = " Write your answer in \\\\boxed{} format."


def map_row(split: str):
    def process_fn(example, idx: int):
        problem = example.get("Problem")
        short_answer = example.get("Short Answer")
        if problem is None or short_answer is None:
            raise ValueError("Expected keys 'Problem' and 'Short Answer' in IMO AnswerBench JSONL row.")

        data = {
            "data_source": "imo_answerbench",
            "prompt": [{"role": "user", "content": problem.strip() + IMO_SUFFIX}],
            "reward_model": {
                "style": "rule",
                "ground_truth": str(short_answer).strip(),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "problem_id": example.get("Problem ID"),
                "category": example.get("Category"),
                "subcategory": example.get("Subcategory"),
                "source": example.get("Source"),
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", default="~/data/imo_llm_judge")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = datasets.load_dataset("json", data_files=args.train_jsonl, split="train", keep_in_memory=True)
    val_dataset = datasets.load_dataset("json", data_files=args.val_jsonl, split="train", keep_in_memory=True)

    train_dataset = train_dataset.map(function=map_row("train"), with_indices=True)
    val_dataset = val_dataset.map(function=map_row("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.output_dir, "validation.parquet"))

