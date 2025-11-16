#!/usr/bin/env python3
"""
Doctor gameplay examples using the verifiers environment.

This script loads doctor_gameplay.jsonl (which have doctoring instructions
in the system prompt for completing failed games) and uses the ConnectionsEnv
to generate completions.

The doctoring instructions are combined with the base system prompt and passed
as the system_prompt to ConnectionsEnv. The dataset keeps the 'prompt' field
as a list of messages, which is valid syntax when are_datasets_raw=False.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from verifiers import setup_logging
from verifiers.types import ClientConfig



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

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "doctored_gameplay.jsonl"
    project_root = Path(__file__).parent.parent.parent

    print(f"Doctoring gameplay examples using verifiers environment...")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {output_dir}")
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

    # Create doctor_results directory for saving results
    results_dir = output_dir / "doctor_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running evaluation with custom environment...")
    print(f"  Results will be saved to: {results_dir}")
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
        results_path=results_dir,
        state_columns=["guess_history", "complete_reason"],
        save_every=20,
    ))

    # The results are saved by evaluate() to results_dir/results.jsonl
    results_file = results_dir / "results.jsonl"
    if results_file.exists():
        print(f"\n✓ Found results: {results_file}")
        print(f"  Copying to: {output_file}")
        shutil.copy2(results_file, output_file)
    else:
        print(f"\n✗ Error: Results file not found at {results_file}")
        sys.exit(1)

    print(f"\n✓ Successfully doctored examples")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()

