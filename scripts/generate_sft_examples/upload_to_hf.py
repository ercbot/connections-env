#!/usr/bin/env python3
"""
Upload SFT gameplay dataset to Hugging Face Hub.

Usage:
    uv run python scripts/generate_sft_examples/upload_to_hf.py
"""

import json
from pathlib import Path

from datasets import Dataset

REPO_NAME = "ericbotti/connections-gameplay-sft"
DEFAULT_INPUT = Path(__file__).parent / "generate_results" / "sft_dataset.jsonl"


def load_dataset(input_file: Path) -> list:
    examples = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def main():
    print(f"Loading dataset from {DEFAULT_INPUT}...")
    examples = load_dataset(DEFAULT_INPUT)
    print(f"Loaded {len(examples)} examples")

    dataset = Dataset.from_list(examples)
    print(f"\nFeatures: {list(dataset.features.keys())}")
    print(f"Size: {len(dataset):,} examples")

    print("\nSample:")
    sample = examples[0]
    for key, val in sample.items():
        if isinstance(val, list):
            print(f"  {key}: [{len(val)} items]")
        else:
            print(f"  {key}: {val}")

    print(f"\nUploading to {REPO_NAME}...")
    commit_message = input("Enter commit message: ").strip()
    if not commit_message:
        print("No commit message provided, exiting.")
        return

    dataset.push_to_hub(REPO_NAME, commit_message=commit_message)
    print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_NAME}")


if __name__ == "__main__":
    main()
