#!/usr/bin/env python3
"""
Filter puzzle rollouts into good and bad examples

Args:
- input_file: Path to the results.jsonl file containing the environment rollouts

Process each puzzle in the input file (example_id):
1. Group all rollouts for this puzzle together
2. Sort rollouts by quality:
   - Highest reward score
   - Tie-breaker: shortest average assistant message length (by tokens)
3. Choose the best rollout which has no invalid guesses and won the game.
    - Save this to the good examples list
    - If no rollouts remain, save the best rollout to the bad examples list.
4. Wrap all parts of the assistant before the <guess> tags in <think> tags.
5. Save the good examples to good_examples.jsonl
6. Save the bad examples to bad_examples.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from transformers import AutoTokenizer

# Import shared utilities
from utils import (
    is_valid_guesses_only,
    is_won,
    calculate_token_metrics,
    wrap_reasoning_in_tags,
    process_rollout,
    MAX_TOTAL_TOKENS,
    MAX_GENERATION_TOKENS,
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


def wrap_reasoning_in_tags(content: str) -> str:
    """
    Wrap all parts of the assistant message before <guess> tags in <think> tags.
    
    Example:
    "Let's think. <guess>[`WORD1`]</guess>" 
    -> "<think>Let's think.</think>\n\n<guess>[`WORD1`]</guess>"
    """
    if "<guess>" not in content:
        # No guess tag, wrap entire content
        return f"<think>{content}</think>"
    
    # Split on <guess> tag
    parts = content.split("<guess>", 1)
    reasoning = parts[0]
    guess_and_rest = parts[1]
    
    # Strip trailing whitespace from reasoning and wrap in redacted_reasoning tags
    reasoning = reasoning.rstrip()
    if reasoning:
        return f"<think>{reasoning}</think>\n\n<guess>{guess_and_rest}"
    else:
        return f"<guess>{guess_and_rest}"


def process_rollout(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a rollout by wrapping assistant reasoning in <think> tags.

    Returns a new dict with the processed completion messages.
    """
    processed = result.copy()
    completion = result.get("completion", [])

    processed_completion = []
    for msg in completion:
        if msg.get("role") == "assistant":
            processed_msg = msg.copy()
            processed_msg["content"] = wrap_reasoning_in_tags(msg["content"])
            processed_completion.append(processed_msg)
        else:
            processed_completion.append(msg)

    processed["completion"] = processed_completion
    return processed


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
        "Token Limit (Total)",
        "Token Limit (Generation)",
    ]

    for reason in reason_order:
        count = rejection_reasons.get(reason, 0)
        percentage = (count / total) if total > 0 else 0.0
        print(f"{reason:<40} {count:<15,} {percentage:<15.1%}")

    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: step_2_filter.py <input_file>")
        print("  input_file: Path to the results.jsonl file containing the environment rollouts")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    print(f"Reading from: {input_file}")
    print("\nLoading tokenizer (PrimeIntellect/Qwen3-4b)...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-4b")

    # Group rollouts by example_id
    print("\nGrouping rollouts by example_id...")
    rollouts_by_example = defaultdict(list)
    total_rollouts = 0

    with open(input_file, "r") as f:
        for line in f:
            total_rollouts += 1
            result = json.loads(line)
            example_id = result.get("example_id")
            if example_id is None:
                print(f"Warning: Skipping result without example_id")
                continue
            rollouts_by_example[example_id].append(result)

    print(f"Found {total_rollouts} total rollouts across {len(rollouts_by_example)} unique puzzles")

    # Process each puzzle
    good_examples = []
    bad_examples = []
    rejection_reasons = {
        "Invalid Guess": 0,
        "Game Lost": 0,
        "Token Limit (Total)": 0,
        "Token Limit (Generation)": 0,
    }

    print("\nProcessing puzzles...")
    for example_id in sorted(rollouts_by_example.keys()):
        rollouts = rollouts_by_example[example_id]

        # Sort rollouts by quality
        sorted_rollouts = sorted(
            rollouts,
            key=lambda r: (-r.get("reward", 0), calculate_avg_assistant_tokens(r, tokenizer))
        )

        # Find the best rollout that has no invalid guesses, won, and is within token limits
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
                # Only process rollout if it passes the first two checks
                processed = process_rollout(rollout)

                # Calculate token metrics once
                total_tokens, max_gen_tokens = calculate_token_metrics(processed, tokenizer)

                # Check token limits
                if total_tokens > MAX_TOTAL_TOKENS:
                    if i == 0:  # Only track rejection reason for the best rollout
                        rejection_reason = "Token Limit (Total)"
                elif max_gen_tokens > MAX_GENERATION_TOKENS:
                    if i == 0:  # Only track rejection reason for the best rollout
                        rejection_reason = "Token Limit (Generation)"
                else:
                    # All checks passed
                    best_valid_winning = processed
                    break

        if best_valid_winning:
            # Add to good examples (already processed)
            good_examples.append(best_valid_winning)
        else:
            # No valid winning rollout, use the best rollout overall
            best_rollout = sorted_rollouts[0]
            # Process the rollout (add think tags)
            processed = process_rollout(best_rollout)
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

    print("\nDone!")


if __name__ == "__main__":
    main()
