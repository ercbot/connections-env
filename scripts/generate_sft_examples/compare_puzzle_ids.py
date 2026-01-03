#!/usr/bin/env python3
"""
Compare puzzle IDs between train_sft dataset and good_examples.jsonl.

1. Get all puzzle_ids from train_sft dataset (HuggingFace)
2. Get all puzzle_ids from good_examples.jsonl
3. Create new_puzzles.txt with puzzle_ids in train_sft but NOT in good_examples.jsonl
4. Create previous_run_good_examples.jsonl filtered to only include puzzle_ids present in train_sft
"""

import json
from pathlib import Path

from datasets import load_dataset
from connections.dataset import prep_dataset
from connections.rulesets import get_ruleset_config


def get_train_sft_puzzle_ids():
    """Load train_sft dataset and extract all puzzle IDs."""
    print("Loading train_sft dataset from HuggingFace...")
    train_sft_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    ruleset_config = get_ruleset_config("nyt")
    train_sft_dataset = prep_dataset(train_sft_dataset, ruleset_config)
    
    # Extract puzzle IDs
    puzzle_ids = set()
    for example in train_sft_dataset:
        puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
        if puzzle_id:
            puzzle_ids.add(puzzle_id)
    
    print(f"  Found {len(puzzle_ids)} puzzle IDs in train_sft dataset")
    return puzzle_ids


def get_good_examples_puzzle_ids(good_examples_path):
    """Load good_examples.jsonl and extract all puzzle IDs."""
    print(f"Loading puzzle IDs from {good_examples_path}...")
    
    puzzle_ids = set()
    with open(good_examples_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
                if puzzle_id:
                    puzzle_ids.add(puzzle_id)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line: {e}")
                continue
    
    print(f"  Found {len(puzzle_ids)} puzzle IDs in input file")
    return puzzle_ids


def filter_good_examples(good_examples_path, train_sft_puzzle_ids, output_path):
    """Filter good_examples.jsonl to only include puzzle_ids present in train_sft."""
    print(f"Filtering {good_examples_path} to only include puzzle_ids in train_sft...")
    
    filtered_count = 0
    total_count = 0
    
    with open(good_examples_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                example = json.loads(line)
                puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
                
                if puzzle_id and puzzle_id in train_sft_puzzle_ids:
                    f_out.write(line + '\n')
                    filtered_count += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line: {e}")
                continue
    
    print(f"  Filtered from {total_count} to {filtered_count} examples")
    return filtered_count


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare puzzle IDs between train_sft dataset and a completed examples file"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("good_examples.jsonl"),
        help="Path to completed examples file (default: good_examples.jsonl in current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input file)"
    )

    args = parser.parse_args()
    good_examples_path = args.input_file

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = good_examples_path.parent

    new_puzzles_path = output_dir / "new_puzzles.txt"
    previous_run_path = output_dir / "previous_run_good_examples.jsonl"

    # Check if input file exists
    if not good_examples_path.exists():
        print(f"Error: {good_examples_path} not found")
        return 1
    
    # Get puzzle IDs from train_sft dataset
    train_sft_puzzle_ids = get_train_sft_puzzle_ids()
    print()
    
    # Get puzzle IDs from input file
    good_examples_puzzle_ids = get_good_examples_puzzle_ids(good_examples_path)
    print()

    # Find puzzle IDs in train_sft but NOT in input file
    new_puzzle_ids = train_sft_puzzle_ids - good_examples_puzzle_ids
    print(f"Puzzle IDs in train_sft but NOT in input file: {len(new_puzzle_ids)}")
    
    # Save new_puzzles.txt
    print(f"\nSaving new puzzle IDs to {new_puzzles_path}...")
    with open(new_puzzles_path, 'w') as f:
        for puzzle_id in sorted(new_puzzle_ids):
            f.write(f"{puzzle_id}\n")
    print(f"  Saved {len(new_puzzle_ids)} puzzle IDs")
    
    # Filter and save previous_run_good_examples.jsonl
    print(f"\nSaving filtered good examples to {previous_run_path}...")
    filtered_count = filter_good_examples(
        good_examples_path,
        train_sft_puzzle_ids,
        previous_run_path
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Train SFT puzzle IDs: {len(train_sft_puzzle_ids)}")
    print(f"  Input file puzzle IDs: {len(good_examples_puzzle_ids)}")
    print(f"  New puzzle IDs (in train_sft, not in input): {len(new_puzzle_ids)}")
    print(f"  Previous run good examples (filtered): {filtered_count}")
    print(f"  Puzzle IDs in input but NOT in train_sft: {len(good_examples_puzzle_ids - train_sft_puzzle_ids)}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

