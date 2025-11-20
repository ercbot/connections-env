#!/usr/bin/env python3
"""
Generate Examples for Supervised Fine-Tuning

1. Load the "train_sft" split from "ericbotti/connections-puzzles" dataset
2. Use the eval function in verifiers to generate examples for supervised fine-tuning
 - load all puzzles in the "train_sft" split
 - run 3 rollouts for each puzzle
 - save the "guess_history" and "completion" state columns
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers import setup_logging
from verifiers.utils.eval_utils import save_results

from connections.environment import ConnectionsEnv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_output_directories, copy_raw_results_to_output, create_client


def load_puzzle_ids(file_path):
    """Load puzzle IDs from a text file (one ID per line)."""
    puzzle_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                puzzle_ids.add(line)
    return puzzle_ids

async def main():
    parser = argparse.ArgumentParser(
        description="Generate examples for supervised fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generate_results"),
        help="Directory to save examples (default: generate_results)",
    )
    parser.add_argument(
        "-r", "--rollouts",
        type=int,
        default=3,
        help="Number of rollouts per example (default: 3)",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to process, -1 for all (default: -1)",
    )
    parser.add_argument(
        "-p", "--puzzle-ids-file",
        type=Path,
        help="Text file containing puzzle IDs to filter (one ID per line)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="deepseek-chat",
        help="Model to use for rollouts (default: deepseek-chat)",
    )
    args = parser.parse_args()
    
    # Setup output directories using unified utility
    output_dir, raw_results_dir, final_output_dir, iteration, output_file = setup_output_directories(
        base_output_dir=args.output_dir,
        results_subdir="",  # No subdirectory for step_1
        filename_pattern="results_{}.jsonl"
    )
    
    rollouts = args.rollouts
    num_examples = args.num_examples
    puzzle_ids_file = args.puzzle_ids_file
    verbose = args.verbose

    # Setup logging
    setup_logging("DEBUG" if verbose else "INFO")

    # Load puzzle IDs if specified
    puzzle_ids = None
    if puzzle_ids_file:
        puzzle_ids_file = puzzle_ids_file.resolve()
        if not puzzle_ids_file.exists():
            print(f"Error: Puzzle IDs file not found: {puzzle_ids_file}")
            sys.exit(1)
        print(f"Loading puzzle IDs from {puzzle_ids_file}...")
        puzzle_ids = load_puzzle_ids(puzzle_ids_file)
        print(f"  Loaded {len(puzzle_ids)} puzzle IDs")
        print()

    
    print(f"Generating SFT examples...")
    print(f"  Output directory: {output_dir}")
    print(f"  Raw results directory: {raw_results_dir}")
    print(f"  Iteration: {iteration}")
    print(f"  Rollouts per example: {rollouts}")
    print(f"  Number of examples: {num_examples if num_examples > 0 else 'all'}")
    if puzzle_ids:
        print(f"  Filtering to {len(puzzle_ids)} puzzle IDs")
    print()

    # Load the train_sft dataset
    print("Loading train_sft dataset...")
    train_sft_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
    
    # Filter by puzzle IDs if specified
    if puzzle_ids:
        print(f"Filtering dataset to {len(puzzle_ids)} puzzle IDs...")
        original_size = len(train_sft_dataset)
        
        def filter_by_puzzle_id(example):
            puzzle_id = str(example.get("puzzle_id", ""))
            return puzzle_id in puzzle_ids
        
        train_sft_dataset = train_sft_dataset.filter(filter_by_puzzle_id)
        filtered_size = len(train_sft_dataset)
        print(f"  Filtered from {original_size} to {filtered_size} examples")
        
        if filtered_size == 0:
            print("  Error: No examples found with the specified puzzle IDs")
            sys.exit(1)
        
        # Check if all requested puzzle IDs were found
        found_ids = set()
        for example in train_sft_dataset:
            puzzle_id = str(example.get("puzzle_id", ""))
            found_ids.add(puzzle_id)
        
        missing_ids = puzzle_ids - found_ids
        if missing_ids:
            print(f"  Warning: {len(missing_ids)} puzzle IDs not found in dataset: {sorted(missing_ids)}")
        print()
    
    # Create environment with train_sft as eval_dataset
    print("Creating ConnectionsEnv...")
    vf_env = ConnectionsEnv(
        ruleset="nyt",
        eval_dataset=train_sft_dataset,
    )
    
    # Create AsyncOpenAI client
    try:
        client = create_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
     
    # Run evaluation
    try:
        results = await vf_env.evaluate(
            client=client,
            model=args.model,
            sampling_args={},
            num_examples=num_examples,
            rollouts_per_example=rollouts,
            max_concurrent=32,
            interleave_scoring=True,
            results_path=raw_results_dir,
            state_columns=["guess_history", "complete_reason"],
            save_every=20,
        )

        print(f"\nCompleted {len(results.completion)} rollouts successfully!")
        
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user. Partial results may be saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error running evaluation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    save_results(results)
    
    # Copy results from raw directory to output directory with iteration number
    print(f"\nCopying results to {output_file}...")
    if copy_raw_results_to_output(raw_results_dir, output_file):
        print(f"✓ Results saved to: {output_file}")
    else:
        print(f"⚠ Warning: Raw results file not found at {raw_results_dir / 'results.jsonl'}")

if __name__ == "__main__":
    asyncio.run(main())
