#!/usr/bin/env python3
"""
Prepare bad examples for retry by truncating to their best checkpoint.

Takes bad examples (games that had invalid guesses or were lost) and truncates
them back to their last valid, recoverable state. This creates a "checkpoint"
the model can continue from naturally in step_4.

Truncation logic:
1. Remove all messages after the first invalid guess
2. If the remaining state would cause game loss (too many mistakes), truncate further
3. Result: a recoverable game state with valid history

For each bad example with gameplay issues:
1. Only process examples with Rejection Reason "invalid guess" or "game lost"
2. Truncate to last valid state
3. Move truncated completion into prompt for continuation
4. Save to doctor_gameplay.jsonl

Outputs stats comparing input vs truncated examples.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import plotext as plt

# Import shared utilities
from utils import (
    truncate_processed_rollout,
)


# Rejection reasons that indicate gameplay issues (only these are handled in this step)
# These match the values in metadata["Rejection Reason"]
GAMEPLAY_REJECTION_REASONS = {"invalid guess", "game lost"}


def create_truncated_example(
    truncated_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a truncated example ready for continuation.

    Takes the truncated rollout and moves the completion into the prompt,
    so the model can continue from the truncation point.
    """
    prompt = truncated_result.get("prompt", [])
    completion = truncated_result.get("completion", [])

    # Append the truncated completion to the prompt
    # This gives the model the conversation context up to the truncation point
    new_prompt = prompt + completion

    # Create the truncated example
    result = truncated_result.copy()
    result["prompt"] = new_prompt
    result["completion"] = []  # Clear completion - will be generated fresh

    # Store the truncated guess_history in info for environment reconstruction
    truncated_guess_history = truncated_result.get("guess_history", [])
    if "info" not in result:
        result["info"] = {}
    result["info"]["resumed_from_guess_history"] = truncated_guess_history

    # Clear guess_history - will be reconstructed from info
    result["guess_history"] = []

    return result


def calculate_stats(examples: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Calculate and print stats for a list of examples."""
    if not examples:
        return {}

    gameplay_messages = []  # Total messages minus initial 2 (system + initial user)
    scores = []
    categories_found = []

    for ex in examples:
        prompt = ex.get("prompt", [])
        completion = ex.get("completion", [])
        
        # Gameplay messages = total - 2 (system + initial user message)
        total_messages = len(prompt) + len(completion)
        gameplay_messages.append(max(0, total_messages - 2))
        
        scores.append(ex.get("reward", 0))
        
        # Count found categories from guess_history
        guess_history = ex.get("guess_history", [])
        found = sum(1 for g in guess_history if g.get("status") in ("correct", "auto"))
        categories_found.append(found)

    def percentiles(values):
        if not values:
            return {}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "min": sorted_vals[0],
            "p25": sorted_vals[n // 4],
            "median": sorted_vals[n // 2],
            "p75": sorted_vals[3 * n // 4],
            "max": sorted_vals[-1],
            "avg": sum(sorted_vals) / n,
        }

    stats = {
        "count": len(examples),
        "gameplay_messages": percentiles(gameplay_messages),
        "scores": percentiles(scores),
        "categories_found": percentiles(categories_found),
    }

    return stats


def print_stats_table(stats: Dict[str, Any], label: str):
    """Print a formatted stats table."""
    if not stats:
        print(f"\n{label}: No examples")
        return

    print(f"\n{'=' * 90}")
    print(f"{label} (N = {stats['count']})")
    print("=" * 90)
    print(f"{'Metric':<25} {'Min':<10} {'25th %':<10} {'Median':<10} {'75th %':<10} {'Max':<10} {'Avg':<10}")
    print("-" * 90)

    for metric_name, metric_key in [
        ("Gameplay Messages", "gameplay_messages"),
        ("Score", "scores"),
        ("Categories Found", "categories_found"),
    ]:
        p = stats.get(metric_key, {})
        if p:
            print(f"{metric_name:<25} {p['min']:<10.2f} {p['p25']:<10.2f} {p['median']:<10.2f} {p['p75']:<10.2f} {p['max']:<10.2f} {p['avg']:<10.2f}")

    print("=" * 90)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare bad examples for doctoring by truncation"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("bad_examples.jsonl"),
        help="Path to the bad_examples.jsonl file (default: bad_examples.jsonl in current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    gameplay_output = output_dir / "doctor_gameplay.jsonl"

    print(f"Reading from: {input_file}")
    print(f"Output: {gameplay_output}")
    print()

    # Load bad examples
    bad_examples = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                bad_examples.append(json.loads(line))

    print(f"Loaded {len(bad_examples)} bad examples")

    # Filter for gameplay rejection reasons only
    gameplay_examples = []
    skipped_by_reason = defaultdict(list)

    for example in bad_examples:
        metadata = example.get("metadata", {})
        rejection_reason = metadata.get("Rejection Reason")

        if rejection_reason in GAMEPLAY_REJECTION_REASONS:
            gameplay_examples.append(example)
        else:
            skipped_by_reason[rejection_reason or "unknown"].append(example)

    print(f"\nFiltered by Rejection Reason:")
    print(f"  Gameplay issues: {len(gameplay_examples)}")
    for reason, examples in sorted(skipped_by_reason.items()):
        print(f"  Skipped ({reason}): {len(examples)}")

    # Print stats for input examples (before truncation)
    print_stats_table(calculate_stats(gameplay_examples, "INPUT"), "INPUT EXAMPLES (before truncation)")

    # Process gameplay examples (truncate only)
    if gameplay_examples:
        print("\nTruncating examples...")
        truncated_examples = []
        turns_remaining_list = []  # Track turns remaining after truncation for histogram

        for i, example in enumerate(gameplay_examples):
            original_guess_history = example.get("guess_history", [])
            
            # Truncate
            truncated = truncate_processed_rollout(example, original_guess_history)
            result = create_truncated_example(truncated)
            
            # Count truncated gameplay messages (in the new prompt, which includes old completion)
            truncated_prompt = result.get("prompt", [])
            truncated_total_messages = len(truncated_prompt)
            truncated_gameplay_messages = max(0, truncated_total_messages - 2)
            
            # Calculate turns remaining (each turn = 2 messages: assistant + user)
            turns_remaining = truncated_gameplay_messages / 2
            turns_remaining_list.append(turns_remaining)
            
            truncated_examples.append(result)

        # Print stats for truncated examples
        # For truncated examples, we need to look at prompt length (which now includes old completion)
        # and the resumed_from_guess_history for categories found
        truncated_stats_examples = []
        for ex in truncated_examples:
            # Create a view with guess_history from resumed_from_guess_history for stats
            view = ex.copy()
            view["guess_history"] = ex.get("info", {}).get("resumed_from_guess_history", [])
            truncated_stats_examples.append(view)

        print_stats_table(calculate_stats(truncated_stats_examples, "OUTPUT"), "OUTPUT EXAMPLES (after truncation)")

        # Generate histogram of turns remaining
        print("\n" + "=" * 90)
        print("TURNS REMAINING AFTER TRUNCATION")
        print("=" * 90)
        
        if turns_remaining_list:
            # Create histogram data
            max_turns = int(max(turns_remaining_list)) + 1
            bins = list(range(0, max_turns + 1))
            
            # Count occurrences for each bin
            counts = [0] * len(bins)
            for turns in turns_remaining_list:
                bin_idx = int(turns)
                if bin_idx < len(counts):
                    counts[bin_idx] += 1
            
            # Plot histogram
            plt.clear_figure()
            plt.simple_bar(bins, counts, width=50, title="Turns Remaining After Truncation")
            plt.xlabel("Number of Turns Remaining")
            plt.ylabel("Count")
            plt.show()
            
            print(f"\nTotal examples: {len(turns_remaining_list)}")
            print(f"Min turns remaining: {min(turns_remaining_list):.1f}")
            print(f"Max turns remaining: {max(turns_remaining_list):.1f}")
            print(f"Average turns remaining: {sum(turns_remaining_list)/len(turns_remaining_list):.1f}")
        
        print("=" * 90)

        # Save results
        print(f"\nSaving {len(truncated_examples)} examples -> {gameplay_output}")
        with open(gameplay_output, "w") as f:
            for example in truncated_examples:
                f.write(json.dumps(example) + "\n")
    else:
        print("\nNo gameplay examples to process")

    print("\nDone!")


if __name__ == "__main__":
    main()
