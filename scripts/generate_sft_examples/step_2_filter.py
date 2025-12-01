#!/usr/bin/env python3
"""
Filter puzzle rollouts into good and bad examples (gameplay failures only)

Args:
- input_dir_or_file (optional): Path to directory containing .jsonl files, or a single .jsonl file
  Defaults to "generate_results" directory if not provided
- --export-rerun-puzzles (optional): Export rerun_puzzles.txt with puzzle IDs that need to be rerun:
  - Puzzle IDs present in bad examples
  - Puzzle IDs present in train_sft dataset but absent from the results loaded in

Process each puzzle in the input files (puzzle_id):
1. Load and combine all .jsonl files from the input directory
2. Group all rollouts for each puzzle together
3. Sort rollouts by quality:
   - Highest reward score
   - Tie-breaker: shortest average assistant message length (by tokens)
4. Choose the best rollout which has no invalid guesses and won the game.
    - Save this to the good examples list
    - If no rollouts remain, save the best rollout to the bad examples list.
5. Wrap all parts of the assistant before the <guess> tags in <think> tags.
6. Save the good examples to good_examples.jsonl
7. Save the bad examples to bad_examples.jsonl
8. If --export-rerun-puzzles is set, export rerun_puzzles.txt with puzzle IDs that need to be rerun:
   - All puzzle IDs from bad examples
   - All puzzle IDs in train_sft dataset that are missing from the results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset

# Import shared utilities
from utils import (
    is_valid_guesses_only,
    is_won,
    process_rollout,
)


def calculate_avg_assistant_tokens(result: Dict[str, Any], tokenizer) -> float:
    """Calculate the average number of tokens in assistant messages."""
    completion = result.get("completion", [])
    assistant_messages = [msg["content"] for msg in completion if msg.get("role") == "assistant"]

    if not assistant_messages:
        return 0.0

    total_tokens = sum(len(tokenizer.encode(msg)) for msg in assistant_messages)
    return total_tokens / len(assistant_messages)


def calculate_rollout_stats(result: Dict[str, Any], tokenizer) -> Dict[str, float]:
    """Calculate statistics for a single rollout."""
    completion = result.get("completion", [])

    # Count assistant turns
    assistant_messages = [msg for msg in completion if msg.get("role") == "assistant"]
    num_turns = len(assistant_messages)

    # Calculate completion tokens (tokens in assistant messages)
    completion_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in assistant_messages)

    # Calculate tokens per completion (average tokens per assistant message)
    tokens_per_completion = completion_tokens / num_turns if num_turns > 0 else 0.0

    # Calculate total tokens (all messages, excluding last if not assistant)
    messages_for_total = completion
    if messages_for_total and messages_for_total[-1].get("role") != "assistant":
        messages_for_total = messages_for_total[:-1]

    total_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in messages_for_total)

    # Get score (reward)
    score = result.get("reward", 0.0)

    return {
        "turns": num_turns,
        "completion_tokens": completion_tokens,
        "tokens_per_completion": tokens_per_completion,
        "total_tokens": total_tokens,
        "score": score,
    }


def select_best_rollout(rollouts: List[Dict[str, Any]], tokenizer) -> Optional[Dict[str, Any]]:
    """
    Select the best rollout from a list.

    Selection criteria:
    1. Highest reward score
    2. Tie-breaker: shortest average assistant message length (by tokens)
    """
    if not rollouts:
        return None

    # Calculate metrics for all rollouts
    rollouts_with_metrics = [
        (rollout, rollout.get("reward", 0), calculate_avg_assistant_tokens(rollout, tokenizer))
        for rollout in rollouts
    ]

    # Sort: highest reward first, then lowest avg tokens
    rollouts_with_metrics.sort(key=lambda x: (-x[1], x[2]))

    return rollouts_with_metrics[0][0]


def print_stats_table(good_examples: List[Dict[str, Any]], tokenizer):
    """Print a statistics table for good examples."""
    if not good_examples:
        print("\nNo good examples to show statistics for.")
        return

    # Calculate stats for all good examples
    all_stats = [calculate_rollout_stats(example, tokenizer) for example in good_examples]

    # Extract each metric into separate lists for percentile calculations
    turns = sorted([s["turns"] for s in all_stats])
    completion_tokens = sorted([s["completion_tokens"] for s in all_stats])
    tokens_per_completion = sorted([s["tokens_per_completion"] for s in all_stats])
    total_tokens = sorted([s["total_tokens"] for s in all_stats])
    scores = sorted([s["score"] for s in all_stats])

    def get_percentiles(values):
        """Calculate min, p25, median, p75, max, and average for a list of values."""
        n = len(values)
        return {
            "min": values[0],
            "p25": values[n // 4],
            "median": values[n // 2],
            "p75": values[3 * n // 4],
            "max": values[-1],
            "avg": sum(values) / n,
        }

    turns_stats = get_percentiles(turns)
    completion_tokens_stats = get_percentiles(completion_tokens)
    tokens_per_completion_stats = get_percentiles(tokens_per_completion)
    total_tokens_stats = get_percentiles(total_tokens)
    score_stats = get_percentiles(scores)

    print("\n" + "=" * 120)
    print(f"GOOD EXAMPLES STATISTICS (N = {len(good_examples)})")
    print("=" * 120)
    print(f"{'Metric':<25} {'Min':<12} {'25th %':<12} {'Median':<12} {'75th %':<12} {'Max':<12} {'Average':<12}")
    print("-" * 120)
    print(f"{'Turns Per Puzzle':<25} {turns_stats['min']:<12.2f} {turns_stats['p25']:<12.2f} {turns_stats['median']:<12.2f} {turns_stats['p75']:<12.2f} {turns_stats['max']:<12.2f} {turns_stats['avg']:<12.2f}")
    print(f"{'Completion Tokens':<25} {completion_tokens_stats['min']:<12.2f} {completion_tokens_stats['p25']:<12.2f} {completion_tokens_stats['median']:<12.2f} {completion_tokens_stats['p75']:<12.2f} {completion_tokens_stats['max']:<12.2f} {completion_tokens_stats['avg']:<12.2f}")
    print(f"{'Tokens / Completion':<25} {tokens_per_completion_stats['min']:<12.2f} {tokens_per_completion_stats['p25']:<12.2f} {tokens_per_completion_stats['median']:<12.2f} {tokens_per_completion_stats['p75']:<12.2f} {tokens_per_completion_stats['max']:<12.2f} {tokens_per_completion_stats['avg']:<12.2f}")
    print(f"{'Total Tokens':<25} {total_tokens_stats['min']:<12.2f} {total_tokens_stats['p25']:<12.2f} {total_tokens_stats['median']:<12.2f} {total_tokens_stats['p75']:<12.2f} {total_tokens_stats['max']:<12.2f} {total_tokens_stats['avg']:<12.2f}")
    print(f"{'Score':<25} {score_stats['min']:<12.2f} {score_stats['p25']:<12.2f} {score_stats['median']:<12.2f} {score_stats['p75']:<12.2f} {score_stats['max']:<12.2f} {score_stats['avg']:<12.2f}")
    print("=" * 120)


def print_rejection_reasons_table(rejection_reasons: Dict[str, int]):
    """Print a table showing rejection reasons for bad examples."""
    if not rejection_reasons:
        print("\nNo bad examples to show rejection reasons for.")
        return

    total = sum(rejection_reasons.values())

    print("\n" + "=" * 80)
    print(f"BAD EXAMPLES REJECTION REASONS (N = {total})")
    print("=" * 80)
    print(f"{'Rejection Reason':<40} {'Count':<15} {'Percentage':<15}")
    print("-" * 80)

    # Define the order of reasons
    reason_order = [
        "Invalid Guess",
        "Game Lost",
    ]

    for reason in reason_order:
        count = rejection_reasons.get(reason, 0)
        percentage = (count / total) if total > 0 else 0.0
        print(f"{reason:<40} {count:<15,} {percentage:<15.1%}")

    print("=" * 80)


def get_train_sft_puzzle_ids() -> Set[str]:
    """Load train_sft dataset and extract all puzzle IDs."""
    print("Loading train_sft dataset from HuggingFace...")
    train_sft_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
    
    # Extract puzzle IDs
    puzzle_ids = set()
    for puzzle in train_sft_dataset:
        puzzle_id = str(puzzle.get("puzzle_id", ""))
        if puzzle_id:
            puzzle_ids.add(puzzle_id)
    
    print(f"  Found {len(puzzle_ids)} puzzle IDs in train_sft dataset")
    return puzzle_ids


def load_and_combine_results(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load and combine all results from a directory or single file.
    
    If input_path is a directory, finds all .jsonl files and combines them.
    If input_path is a file, loads just that file.
    """
    all_results = []
    
    if input_path.is_file():
        # Single file
        print(f"Loading from file: {input_path}")
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))
        print(f"  Loaded {len(all_results)} results")
    elif input_path.is_dir():
        # Directory - find all .jsonl files
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"Error: No .jsonl files found in directory {input_path}")
            sys.exit(1)
        
        print(f"Found {len(jsonl_files)} .jsonl file(s) in directory:")
        for f in jsonl_files:
            print(f"  - {f.name}")
        
        # Load all examples from all files
        for file_path in jsonl_files:
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        
        print(f"  Loaded {len(all_results)} total results")
    else:
        print(f"Error: Input path {input_path} does not exist or is not a file/directory")
        sys.exit(1)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Filter puzzle rollouts into good and bad examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir_or_file",
        type=Path,
        nargs="?",
        default=Path("generate_results"),
        help="Path to directory containing .jsonl files, or a single .jsonl file (default: generate_results)",
    )
    parser.add_argument(
        "--export-rerun-puzzles",
        action="store_true",
        help="Export rerun_puzzles.txt with puzzle IDs from train_sft dataset that are not in results",
    )
    
    args = parser.parse_args()
    input_path = args.input_dir_or_file

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    print(f"Reading from: {input_path}")
    print("\nLoading tokenizer (PrimeIntellect/Qwen3-4b)...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-4b")

    # Load and combine all results
    all_results = load_and_combine_results(input_path)

    # Group rollouts by puzzle_id
    print("\nGrouping rollouts by puzzle_id...")
    rollouts_by_puzzle = defaultdict(list)
    total_rollouts = 0

    for result in all_results:
        total_rollouts += 1
        puzzle_id = str(result.get("info", {}).get("puzzle_id", ""))
        if not puzzle_id:
            print(f"Warning: Skipping result without puzzle_id")
            continue
        rollouts_by_puzzle[puzzle_id].append(result)

    print(f"Found {total_rollouts} total rollouts across {len(rollouts_by_puzzle)} unique puzzles")

    # Process each puzzle
    good_examples = []
    bad_examples = []
    rejection_reasons = {
        "Invalid Guess": 0,
        "Game Lost": 0,
    }

    print("\nProcessing puzzles...")
    for puzzle_id in sorted(rollouts_by_puzzle.keys()):
        rollouts = rollouts_by_puzzle[puzzle_id]

        # Sort rollouts by quality
        sorted_rollouts = sorted(
            rollouts,
            key=lambda r: (-r.get("reward", 0), calculate_avg_assistant_tokens(r, tokenizer))
        )

        # Find the best rollout that has no invalid guesses and won
        best_valid_winning = None
        rejection_reason = None
        for i, rollout in enumerate(sorted_rollouts):
            # Check non-processed criteria first (faster)
            if not is_valid_guesses_only(rollout):
                if i == 0:  # Only track rejection reason for the best rollout
                    rejection_reason = "Invalid Guess"
            elif not is_won(rollout):
                if i == 0:  # Only track rejection reason for the best rollout
                    rejection_reason = "Game Lost"
            else:
                # All gameplay checks passed - process and accept
                processed = process_rollout(rollout)
                best_valid_winning = processed
                break

        if best_valid_winning:
            # Add to good examples (already processed)
            best_valid_winning["tags"] = ["good example"]
            best_valid_winning["metadata"] = {
                "Quality": "good",
                "Complete Reason": best_valid_winning.get("complete_reason", "unknown"),
            }
            good_examples.append(best_valid_winning)
        else:
            # No valid winning rollout, use the best rollout overall
            best_rollout = sorted_rollouts[0]
            # Process the rollout (add think tags)
            processed = process_rollout(best_rollout)
            # Add tags and metadata based on rejection reason
            if rejection_reason == "Invalid Guess":
                processed["tags"] = ["rr: invalid guess"]
                processed["metadata"] = {
                    "Rejection Reason": "invalid guess",
                    "Complete Reason": processed.get("complete_reason", "unknown"),
                }
            elif rejection_reason == "Game Lost":
                processed["tags"] = ["rr: game lost"]
                processed["metadata"] = {
                    "Rejection Reason": "game lost",
                    "Complete Reason": processed.get("complete_reason", "unknown"),
                }
            else:
                processed["tags"] = ["rr: unknown"]
                processed["metadata"] = {
                    "Rejection Reason": "unknown",
                    "Complete Reason": processed.get("complete_reason", "unknown"),
                }
            bad_examples.append(processed)
            # Track the rejection reason
            if rejection_reason:
                rejection_reasons[rejection_reason] += 1

    # Save results
    good_file = "good_examples.jsonl"
    bad_file = "bad_examples.jsonl"

    print(f"\nSaving results...")
    print(f"  Good examples: {len(good_examples)} -> {good_file}")
    print(f"  Bad examples: {len(bad_examples)} -> {bad_file}")

    with open(good_file, "w") as f:
        for example in good_examples:
            f.write(json.dumps(example) + "\n")

    with open(bad_file, "w") as f:
        for example in bad_examples:
            f.write(json.dumps(example) + "\n")

    # Print statistics table for good examples
    print_stats_table(good_examples, tokenizer)

    # Print rejection reasons table for bad examples
    print_rejection_reasons_table(rejection_reasons)

    # Export rerun_puzzles.txt if requested
    if args.export_rerun_puzzles:
        print("\n" + "=" * 80)
        print("Exporting rerun_puzzles.txt...")
        print("=" * 80)
        
        # Get all puzzle IDs from results (both good and bad)
        result_puzzle_ids = set()
        for example in good_examples:
            puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
            if puzzle_id:
                result_puzzle_ids.add(puzzle_id)
        for example in bad_examples:
            puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
            if puzzle_id:
                result_puzzle_ids.add(puzzle_id)
        
        print(f"  Puzzle IDs in results: {len(result_puzzle_ids)}")
        
        # Get puzzle IDs from bad examples (these need to be rerun)
        bad_puzzle_ids = set()
        for example in bad_examples:
            puzzle_id = str(example.get("info", {}).get("puzzle_id", ""))
            if puzzle_id:
                bad_puzzle_ids.add(puzzle_id)
        
        print(f"  Puzzle IDs in bad examples: {len(bad_puzzle_ids)}")
        
        # Get puzzle IDs from train_sft dataset
        train_sft_puzzle_ids = get_train_sft_puzzle_ids()
        
        # Find puzzle IDs in train_sft but NOT in results (missing from results)
        missing_from_results = train_sft_puzzle_ids - result_puzzle_ids
        
        # Rerun puzzle IDs = bad examples OR missing from results
        rerun_puzzle_ids = bad_puzzle_ids | missing_from_results
        
        # Save rerun_puzzles.txt
        rerun_file = "rerun_puzzles.txt"
        print(f"\nSaving puzzle IDs to rerun to {rerun_file}...")
        with open(rerun_file, "w") as f:
            for puzzle_id in sorted(rerun_puzzle_ids):
                f.write(f"{puzzle_id}\n")
        
        print(f"  Saved {len(rerun_puzzle_ids)} puzzle IDs")
        print(f"  Summary:")
        print(f"    Train SFT puzzle IDs: {len(train_sft_puzzle_ids)}")
        print(f"    Puzzle IDs in results: {len(result_puzzle_ids)}")
        print(f"    Puzzle IDs in bad examples: {len(bad_puzzle_ids)}")
        print(f"    Puzzle IDs missing from results: {len(missing_from_results)}")
        print(f"    Puzzle IDs to rerun (bad + missing): {len(rerun_puzzle_ids)}")
        print(f"    Puzzle IDs in results but NOT in train_sft: {len(result_puzzle_ids - train_sft_puzzle_ids)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
