#!/usr/bin/env python3
"""
Retry failed gameplay examples from their best checkpoint.

This script loads doctor_gameplay.jsonl (truncated examples from step_3) and uses
the ConnectionsEnv to continue the game from the truncation point. The goal is to
get more natural, successful completions by giving the model another chance.

Strategy:
- Step 3 truncated bad examples back to their last valid state (before invalid guesses
  or game-losing mistakes)
- This script continues from that checkpoint, letting the model try again naturally
- No doctoring instructions - the model plays authentically
- Multiple reruns can squeeze more good examples from the pool

After running the environment, it validates the results:
- Valid examples (no invalid guesses, game won) → doctored_examples_N.jsonl
- Invalid examples (still have issues) → doctor_gameplay_rerun.jsonl (for retry)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from datasets import Dataset, load_dataset
from verifiers import setup_logging
from verifiers.types import ClientConfig

# Import shared utilities for validation
sys.path.insert(0, str(Path(__file__).parent))
from utils import is_valid_guesses_only, is_won, process_rollout, setup_output_directories



def load_and_prepare_dataset(doctor_ready_file: Path) -> Dataset:
    """
    Load doctor_gameplay.jsonl as a dataset and prepare it for the environment.

    Loads the file directly as a HuggingFace Dataset and drops columns that
    aren't needed by the environment (like completion, guess_history, reward, etc.).
    Keeps only: prompt, info, and example_id (if present).

    Returns:
        Dataset with only the necessary columns
    """
    # Load directly as HuggingFace Dataset
    dataset = load_dataset("json", data_files=str(doctor_ready_file))["train"]

    # Identify columns to keep
    columns_to_keep = ["prompt", "info"]
    if "example_id" in dataset.column_names:
        columns_to_keep.append("example_id")

    # Drop all other columns
    columns_to_drop = [col for col in dataset.column_names if col not in columns_to_keep]
    if columns_to_drop:
        dataset = dataset.remove_columns(columns_to_drop)

    return dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Doctor gameplay examples using the verifiers environment"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("doctor_gameplay.jsonl"),
        help="Path to doctor_gameplay.jsonl (default: doctor_gameplay.jsonl in current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save results (default: current directory)"
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Number of rollouts per example (default: 1)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    rollouts = args.rollouts
    verbose = args.verbose

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    # Check for required environment variables
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)

    # Setup logging
    setup_logging("DEBUG" if verbose else "INFO")

    # Setup output directories using unified utility
    base_output_dir, raw_results_dir, doctor_results_dir, iteration, output_file = setup_output_directories(
        base_output_dir=output_dir,
        results_subdir="doctor_results",
        filename_pattern="doctored_examples_{}.jsonl"
    )

    print(f"Doctoring gameplay examples using verifiers environment...")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {base_output_dir}")
    print(f"  Doctor results directory: {doctor_results_dir}")
    print(f"  Iteration: {iteration}")
    print(f"  Rollouts per example: {rollouts}")
    print(f"  Model: deepseek-chat")
    print()

    # Load doctor_gameplay.jsonl directly as dataset
    print("Loading doctor_gameplay examples...")
    doctor_dataset = load_and_prepare_dataset(input_file)
    print(f"  Loaded {len(doctor_dataset)} examples")
    print(f"  Dataset columns: {doctor_dataset.column_names}")
    print()

    # Configure client
    client_config = ClientConfig(
        api_key_var="DEEPSEEK_API_KEY",
        api_base_url="https://api.deepseek.com",
        extra_headers={},
    )
    
    # Use the standard ConnectionsEnv - it accepts pre-formatted datasets directly
    print("Creating environment for doctoring...")
    from connections.environment import ConnectionsEnv
    from verifiers.utils.client_utils import setup_client
    
    # Create the environment with our pre-formatted dataset
    # The dataset has 'prompt' field (list of messages) which is valid when are_datasets_raw=False
    # format_dataset will use the existing 'prompt' column as-is (it already has system + user messages)
    # Since prompt already exists, the system_prompt parameter is ignored by format_dataset
    # Set are_datasets_raw=False to skip prep_dataset (which expects raw format)
    doctor_env = ConnectionsEnv(
        ruleset="nyt",
        dataset=doctor_dataset,
        eval_dataset=doctor_dataset,
        are_datasets_raw=False,  # Dataset is already formatted, don't call prep_dataset
        system_prompt=None,  # Ignored since prompt field already exists in dataset
    )
    
    # Setup client
    client = setup_client(client_config)

    print(f"Running evaluation with custom environment...")
    print(f"  Raw results will be saved to: {raw_results_dir}")
    print()
    
    # Run evaluation directly
    results = asyncio.run(doctor_env.evaluate(
        client=client,
        model="deepseek-chat",
        sampling_args={},
        num_examples=-1,
        rollouts_per_example=rollouts,
        max_concurrent=32,
        max_concurrent_generation=None,
        max_concurrent_scoring=None,
        interleave_scoring=True,
        results_path=raw_results_dir,
        state_columns=["guess_history", "complete_reason"],
        save_every=20,
    ))

    # The results are saved by evaluate() to raw_results_dir/results.jsonl
    results_file = raw_results_dir / "results.jsonl"
    if not results_file.exists():
        print(f"\n✗ Error: Results file not found at {results_file}")
        sys.exit(1)

    print(f"\n✓ Found results: {results_file}")
    print(f"  Validating doctored examples...")

    # Load and validate results
    doctored_results = []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                doctored_results.append(json.loads(line))

    # Load original doctor_gameplay.jsonl to create rerun file
    original_doctor_ready = {}
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                puzzle_id = example.get("info", {}).get("puzzle_id")
                if puzzle_id:
                    original_doctor_ready[puzzle_id] = example

    print(f"  Total input examples: {len(original_doctor_ready)}")
    print(f"  Results from evaluation: {len(doctored_results)}")
    
    # Track which puzzles were processed and which passed
    processed_puzzle_ids = set()
    valid_puzzle_ids = set()
    
    # Validate and separate results
    valid_examples = []
    invalid_examples = []
    missing_puzzle_ids = []  # Track puzzle IDs not found in original input
    rejection_reasons = {
        "Invalid Guess": 0,
        "Game Lost": 0,
        "Not Processed": 0,
    }

    for result in doctored_results:
        puzzle_id = result.get("info", {}).get("puzzle_id")
        if puzzle_id:
            processed_puzzle_ids.add(puzzle_id)

        # Check for None guess_history first
        guess_history = result.get("guess_history")
        if guess_history is None:
            rejection_reasons["Game Lost"] += 1  # Treat None as failed game
            if puzzle_id in original_doctor_ready:
                invalid_examples.append(original_doctor_ready[puzzle_id])
            else:
                missing_puzzle_ids.append(puzzle_id)
            continue

        # Validate gameplay
        if not is_valid_guesses_only(result):
            rejection_reasons["Invalid Guess"] += 1
            # Add original doctor-ready example to rerun list
            if puzzle_id in original_doctor_ready:
                invalid_examples.append(original_doctor_ready[puzzle_id])
            else:
                missing_puzzle_ids.append(puzzle_id)
        elif not is_won(result):
            rejection_reasons["Game Lost"] += 1
            # Add original doctor-ready example to rerun list
            if puzzle_id in original_doctor_ready:
                invalid_examples.append(original_doctor_ready[puzzle_id])
            else:
                missing_puzzle_ids.append(puzzle_id)
        else:
            # Process and add to valid examples
            processed = process_rollout(result)
            valid_examples.append(processed)
            if puzzle_id:
                valid_puzzle_ids.add(puzzle_id)
    
    # Add all unprocessed examples to rerun list
    unprocessed_count = 0
    for puzzle_id, example in original_doctor_ready.items():
        if puzzle_id not in processed_puzzle_ids:
            invalid_examples.append(example)
            unprocessed_count += 1
    
    if unprocessed_count > 0:
        rejection_reasons["Not Processed"] = unprocessed_count

    # Save valid doctored examples
    print(f"\n  Valid examples: {len(valid_examples)}")
    print(f"  Invalid/Incomplete examples: {len(invalid_examples)}")
    print(f"    - Invalid Guess: {rejection_reasons['Invalid Guess']}")
    print(f"    - Game Lost: {rejection_reasons['Game Lost']}")
    print(f"    - Not Processed: {rejection_reasons.get('Not Processed', 0)}")
    
    if missing_puzzle_ids:
        print(f"\n  ⚠ Warning: {len(missing_puzzle_ids)} invalid examples had puzzle_ids not found in input file")
        print(f"    These examples cannot be rerun: {missing_puzzle_ids}")

    if valid_examples:
        print(f"\n  Saving valid examples to: {output_file}")
        with open(output_file, "w") as f:
            for example in valid_examples:
                f.write(json.dumps(example) + "\n")
        print(f"  ✓ Saved {len(valid_examples)} valid examples")

    # Save invalid examples for rerun (pass-through original doctor-ready format)
    rerun_file = base_output_dir / "doctor_gameplay_rerun.jsonl"
    if invalid_examples:
        print(f"\n  Saving examples for rerun to: {rerun_file}")
        with open(rerun_file, "w") as f:
            for example in invalid_examples:
                f.write(json.dumps(example) + "\n")
        print(f"  ✓ Saved {len(invalid_examples)} examples to rerun file")
        print(f"\n  ⚠ {len(invalid_examples)} examples need to be rerun")
        print(f"  To rerun: uv run step_4_doctor_gameplay.py {rerun_file} --output-dir {base_output_dir}")
    else:
        # Create empty rerun file to signal completion
        print(f"\n  Creating empty rerun file (all examples passed): {rerun_file}")
        with open(rerun_file, "w") as f:
            pass
        print(f"  ✓ All examples passed validation!")

    print(f"\n✓ Successfully doctored {len(valid_examples)}/{len(doctored_results)} examples")
    if valid_examples:
        print(f"  Valid examples saved to: {output_file}")
    print(f"  Rerun file: {rerun_file} ({len(invalid_examples)} examples)")

    print(f"\nNext steps:")
    if invalid_examples:
        print(f"  1. Rerun failed examples: uv run step_4_doctor_gameplay.py {rerun_file}")
        print(f"  2. Then proceed: uv run step_5_reduce_tokens.py")
    else:
        print(f"  Proceed: uv run step_5_reduce_tokens.py")


if __name__ == "__main__":
    main()

