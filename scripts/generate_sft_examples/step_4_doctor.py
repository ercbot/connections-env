#!/usr/bin/env python3
"""
Doctor examples using the verifiers environment.

This script loads doctor_ready_examples.jsonl (which have doctoring instructions
in the system prompt) and uses the ConnectionsEnv to generate completions.

The doctoring instructions are combined with the base system prompt and passed
as the system_prompt to ConnectionsEnv. The dataset keeps the 'prompt' field
as a list of messages, which is valid syntax when are_datasets_raw=False.
"""

import asyncio
import json
import os
import sys
import glob
import shutil
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset, load_dataset
from verifiers import setup_logging
from verifiers.types import ClientConfig, EvalConfig



def convert_doctor_ready_to_dataset(doctor_ready_file: Path) -> tuple[Dataset, str | None]:
    """
    Convert doctor_ready_examples.jsonl to a Dataset format for the environment.
    
    The doctor_ready_examples have doctoring instructions in the system prompt.
    We extract the system prompt (which combines base + doctoring instructions) and
    keep only user messages in the prompt field. The system_prompt will be passed
    separately to ConnectionsEnv.
    
    Returns:
        tuple: (Dataset, system_prompt) where system_prompt is extracted from the first example
    """
    examples = []
    system_prompt = None
    
    with open(doctor_ready_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            
            # Get the prompt messages (already has system + user)
            prompt = example.get("prompt", [])
            
            # Extract system prompt from first example (all should have the same)
            # The system prompt already combines base + doctoring instructions
            if system_prompt is None:
                for msg in prompt:
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                        break
            
            # Keep the full prompt as-is (system + user messages)
            # format_dataset will use this prompt as-is when are_datasets_raw=False
            # Since prompt already exists, format_dataset won't add system_prompt again
            dataset_example = {
                "prompt": prompt,  # List of messages (system + user) - valid when are_datasets_raw=False
                "info": example.get("info", {}),
            }
            
            # Add example_id if present (optional field)
            if "example_id" in example:
                dataset_example["example_id"] = example["example_id"]
            
            examples.append(dataset_example)
    
    return Dataset.from_list(examples), system_prompt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Doctor examples using the verifiers environment"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("doctor_ready_examples.jsonl"),
        help="Path to doctor_ready_examples.jsonl (default: doctor_ready_examples.jsonl in current directory)"
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
    output_file = output_dir / "doctored_examples.jsonl"
    project_root = Path(__file__).parent.parent.parent

    print(f"Doctoring examples using verifiers environment...")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Rollouts per example: {rollouts}")
    print(f"  Model: deepseek-chat")
    print()

    # Load and convert doctor_ready_examples to dataset
    print("Loading and converting doctor_ready_examples...")
    dataset, doctoring_system_prompt = convert_doctor_ready_to_dataset(input_file)
    print(f"  Loaded {len(dataset)} examples")
    if doctoring_system_prompt:
        print(f"  Extracted system prompt with doctoring instructions ({len(doctoring_system_prompt)} chars)")
    print()

    # Save dataset temporarily as JSONL for the environment to load
    temp_dataset_file = output_dir / "temp_doctor_ready_dataset.jsonl"
    with open(temp_dataset_file, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    
    print(f"  Saved temporary dataset to: {temp_dataset_file}")
    print()

    # Configure client
    client_config = ClientConfig(
        api_key_var="DEEPSEEK_API_KEY",
        api_base_url="https://api.deepseek.com",
        extra_headers={},
    )

    # Load dataset as HuggingFace Dataset
    doctor_dataset = load_dataset("json", data_files=str(temp_dataset_file))["train"]
    
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
        max_turns=20,
    )
    
    # Setup client
    client = setup_client(client_config)
    
    # Create a temporary eval_config for getting results path
    temp_eval_config = EvalConfig(
        env_id="connections",
        env_args={},
        env_dir_path=str(project_root),
        model="deepseek-chat",
        client_config=client_config,
        sampling_args={},
        num_examples=-1,
        rollouts_per_example=rollouts,
        max_concurrent=32,
        max_concurrent_generation=None,
        max_concurrent_scoring=None,
        interleave_scoring=True,
        print_results=True,
        verbose=verbose,
        state_columns=["guess_history", "complete_reason"],
        save_results=True,
        save_every=20,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )
    
    # Get results path
    from verifiers.utils.path_utils import get_eval_results_path
    results_path = get_eval_results_path(temp_eval_config)
    
    print(f"Running evaluation with custom environment...")
    print(f"  Results will be saved to: {results_path}")
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
        results_path=results_path,
        state_columns=["guess_history", "complete_reason"],
        save_every=20,
    ))
    
    # The results are already saved by evaluate(), so we just need to copy them
    if results_path and results_path.exists():
        print(f"\n✓ Found results: {results_path}")
        print(f"  Copying to: {output_file}")
        shutil.copy2(results_path, output_file)
    else:
        print(f"\n✗ Error: Results file not found at {results_path}")
        sys.exit(1)
    
    # Clean up temp file
    if temp_dataset_file.exists():
        temp_dataset_file.unlink()
        print(f"  Cleaned up temporary dataset file")
    
    print(f"\n✓ Successfully doctored examples")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()

