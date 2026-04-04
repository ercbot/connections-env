#!/usr/bin/env python3
"""
Upload SFT gameplay dataset to Hugging Face Hub.

Usage:
    uv run python scripts/generate_sft_examples/upload_to_hf.py
"""

import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import DatasetCard

REPO_NAME = "ericbotti/connections-gameplay-sft"
DEFAULT_INPUT = Path(__file__).parent / "generate_results" / "sft_dataset.jsonl"
README_TEMPLATE = Path(__file__).parent / "README.template.md"


def load_dataset(input_file: Path) -> list:
    examples = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def render_readme(examples: list, input_file: Path) -> str:
    template = README_TEMPLATE.read_text()

    count = len(examples)
    accuracies = [e["accuracy"] for e in examples]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    perfect_pct = sum(1 for a in accuracies if a == 1.0) / len(accuracies) * 100 if accuracies else 0.0
    file_size = input_file.stat().st_size

    return template.format(
        train_count=count,
        train_num_bytes=file_size,
        download_size=file_size,
        dataset_size=file_size,
        total_count=count,
        avg_accuracy=avg_accuracy,
        perfect_pct=perfect_pct,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print README and stats without uploading")
    args = parser.parse_args()

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

    readme_content = render_readme(examples, DEFAULT_INPUT)

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN — README output:")
        print("="*60)
        print(readme_content)
        print("="*60)
        print("(Nothing uploaded)")
        return

    print(f"\nUploading to {REPO_NAME}...")
    commit_message = input("Enter commit message: ").strip()
    if not commit_message:
        print("No commit message provided, exiting.")
        return

    dataset.push_to_hub(REPO_NAME, commit_message=commit_message)

    print("Rendering and uploading README...")
    card = DatasetCard(readme_content)
    card.push_to_hub(REPO_NAME)

    print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_NAME}")


if __name__ == "__main__":
    main()
