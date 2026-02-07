#!/usr/bin/env python3
"""
Collate final SFT dataset from iterative generation results.

Loads all results, selects the best rollout per puzzle, filters to good examples
(won + valid guesses + within token limits), validates, and outputs the final dataset.

Usage:
    python collate.py
    python collate.py --output sft_examples.jsonl
    python collate.py --view  # Start web viewer
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from transformers import AutoTokenizer

from iterative_generate import load_existing_results, select_best_rollouts
from token_counting_utils import (
    TOKENIZER_NAME,
    MAX_GENERATION_TOKENS,
    MAX_TOTAL_TOKENS,
    calculate_max_cumulative_tokens,
    mark_over_limit_as_invalid,
)
from utils import is_valid_guesses_only, is_won, process_rollout


def calculate_accuracy(guess_history: list) -> float:
    """Calculate accuracy as correct_guesses / total_guesses."""
    if not guess_history:
        return 0.0
    correct = sum(1 for g in guess_history if g.get("status") in ("correct", "auto"))
    return correct / len(guess_history) if guess_history else 0.0


def validate_example(example: Dict[str, Any], tokenizer, token_limit: int) -> List[str]:
    """
    Run validation checks on a single processed example.

    Returns a list of issues found (empty = passed).
    """
    issues = []

    prompt = example.get("prompt", [])
    completion = example.get("completion", [])
    guess_history = example.get("guess_history", [])
    info = example.get("info", {})

    # Check required fields
    if not prompt:
        issues.append("empty prompt")
    if not completion:
        issues.append("empty completion")
    if not info.get("puzzle_id"):
        issues.append("missing puzzle_id")
    if not info.get("categories"):
        issues.append("missing categories")

    # Check that first message is system prompt
    if prompt and prompt[0].get("role") != "system":
        issues.append(f"first prompt message is '{prompt[0].get('role')}', expected 'system'")

    # Check message roles alternate properly in completion
    for i, msg in enumerate(completion):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            issues.append(f"completion[{i}] has unexpected role '{role}'")
        if not msg.get("content", "").strip():
            issues.append(f"completion[{i}] ({role}) has empty content")

    # Check assistant messages have <guess> tags (in completion only)
    completion_assistant_msgs = [m for m in completion if m.get("role") == "assistant"]
    for i, msg in enumerate(completion_assistant_msgs):
        content = msg.get("content", "")
        if "<guess>" not in content:
            issues.append(f"completion assistant message {i} missing <guess> tag")

    # Check guess count matches assistant message count across prompt + completion
    # (for resumed/doctored rollouts, earlier guesses are in prompt)
    all_messages = prompt + completion
    all_assistant_msgs = [m for m in all_messages if m.get("role") == "assistant"]
    non_auto_guesses = [g for g in guess_history if g.get("status") != "auto"]
    if len(all_assistant_msgs) != len(non_auto_guesses):
        issues.append(
            f"total assistant message count ({len(all_assistant_msgs)}) != "
            f"non-auto guess count ({len(non_auto_guesses)})"
        )

    # Check per-message token limits (completion messages only)
    for i, msg in enumerate(completion_assistant_msgs):
        tokens = len(tokenizer.encode(msg.get("content", "")))
        if tokens > token_limit:
            issues.append(f"completion assistant message {i}: {tokens} tokens > {token_limit} limit")

    # Check cumulative token limit
    max_cumulative = calculate_max_cumulative_tokens(example, tokenizer)
    if max_cumulative > MAX_TOTAL_TOKENS:
        issues.append(f"cumulative tokens {max_cumulative} > {MAX_TOTAL_TOKENS} limit")

    return issues


def collate(
    results_dir: Path,
    tokenizer,
    token_limit: int,
    min_salvage_quality: float,
) -> List[Dict[str, Any]]:
    """
    Load results, select best rollouts, filter to good examples, validate.

    Returns list of validated good examples ready for the final dataset.
    """
    # Load all results
    all_results = load_existing_results(results_dir)
    if not all_results:
        print("No results found!")
        return []

    # Select best rollout per puzzle
    best_rollouts = select_best_rollouts(all_results, min_salvage_quality, tokenizer)
    print(f"Best rollouts: {len(best_rollouts)} unique puzzles")

    # Process and filter
    good_examples = []
    bad_count = 0

    for puzzle_id, rollout in best_rollouts.items():
        processed = process_rollout(rollout)

        # Mark over-limit messages as invalid
        processed["info"]["original_guess_history"] = processed.get("guess_history", []).copy()
        processed = mark_over_limit_as_invalid(processed, tokenizer, token_limit, MAX_TOTAL_TOKENS)

        if is_valid_guesses_only(processed) and is_won(processed):
            good_examples.append(processed)
        else:
            bad_count += 1

    print(f"Good examples: {len(good_examples)}")
    print(f"Bad/over-limit: {bad_count}")

    # Validate all good examples
    print(f"\nValidating {len(good_examples)} examples...")
    validated = []
    validation_failures = 0

    for example in good_examples:
        issues = validate_example(example, tokenizer, token_limit)
        if issues:
            puzzle_id = example.get("info", {}).get("puzzle_id", "?")
            print(f"  FAIL puzzle {puzzle_id}: {'; '.join(issues)}")
            validation_failures += 1
        else:
            validated.append(example)

    if validation_failures:
        print(f"\n{validation_failures} examples failed validation and were excluded")
    else:
        print("All examples passed validation")

    return validated


def synthesize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a processed example to the final dataset format."""
    guess_history = example.get("guess_history", [])
    info = example.get("info", {})

    return {
        "puzzle_id": info.get("puzzle_id", "unknown"),
        "prompt": example.get("prompt", []),
        "completion": example.get("completion", []),
        "accuracy": calculate_accuracy(guess_history),
        "guess_history": guess_history,
        "categories": info.get("categories", []),
        "complete_reason": example.get("complete_reason", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collate and validate final SFT dataset from iterative generation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing results files (default: generate_results/ in script directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: generate_results/sft_dataset.jsonl)"
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=MAX_GENERATION_TOKENS,
        help=f"Max tokens per assistant message. Default: {MAX_GENERATION_TOKENS}"
    )
    parser.add_argument(
        "--min-salvage-quality",
        type=float,
        default=0.0,
        help="Minimum salvage quality score. Default: 0.0"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Start web server to view results after collation"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web viewer (default: 8000)"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    if args.results_dir is None:
        results_dir = script_dir / "generate_results"
    else:
        results_dir = Path(args.results_dir)

    if args.output is None:
        output_file = results_dir / "sft_dataset.jsonl"
    else:
        output_file = args.output

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer ({TOKENIZER_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Token limits: {args.token_limit} per message, {MAX_TOTAL_TOKENS} cumulative")
    print()

    # Collate and validate
    validated_examples = collate(
        results_dir, tokenizer, args.token_limit, args.min_salvage_quality
    )

    if not validated_examples:
        print("\nNo valid examples to output!")
        sys.exit(1)

    # Synthesize final format
    final_examples = [synthesize_example(ex) for ex in validated_examples]

    # Statistics
    accuracies = [e["accuracy"] for e in final_examples]
    perfect = sum(1 for a in accuracies if a == 1.0)

    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Total examples: {len(final_examples)}")
    print(f"  Average accuracy: {sum(accuracies)/len(accuracies):.2%}")
    print(f"  Perfect accuracy: {perfect} ({perfect/len(final_examples)*100:.1f}%)")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in final_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n  Output: {output_file}")

    # Optional viewer
    if args.view:
        # Import viewer from old step_6 if available
        try:
            from step_6_collate_validate import start_viewer_server
            start_viewer_server(final_examples, port=args.port)
        except ImportError:
            print("Viewer not available (step_6_collate_validate.py not found)")


if __name__ == "__main__":
    main()
