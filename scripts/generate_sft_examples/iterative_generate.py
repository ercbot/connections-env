#!/usr/bin/env python3
"""
Iterative SFT Example Generation Script

This script combines the functionality of steps 2, 3, and 4 into a single iterative process:
1. Prep: Load existing results, filter good/bad, prepare salvageable examples for doctoring
2. Generate: Run evaluation on salvageable examples + missing puzzles
3. Repeat for N loops, accumulating results and improving coverage

Usage:
    python iterative_generate.py --loop 5
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from datasets import Dataset, load_dataset

from connections.environment import ConnectionsEnv
from utils import (
    create_client,
    is_valid_guesses_only,
    is_won,
    process_rollout,
    truncate_processed_rollout,
)


def calculate_avg_assistant_tokens(result: Dict) -> float:
    """
    Calculate a simple proxy for average assistant message length.
    Uses character count instead of tokens for simplicity.
    """
    completion = result.get("completion", [])
    assistant_messages = [msg["content"] for msg in completion if msg.get("role") == "assistant"]

    if not assistant_messages:
        return 0.0

    total_chars = sum(len(msg) for msg in assistant_messages)
    # Rough approximation: 1 token ≈ 4 characters
    return (total_chars / 4) / len(assistant_messages)


def calculate_salvage_quality(result: Dict) -> float:
    """
    Calculate salvage quality score with exponentially decaying weights by position.
    Higher score = better for salvaging (correct guesses earlier are worth more).

    Scoring:
    - Correct/auto guess at index i: +1/2^i
    - Mistake (incorrect/one_away) number m: -1/2^(3-m)
      - 1st mistake: -0.125
      - 2nd mistake: -0.25
      - 3rd mistake: -0.5
      - 4th mistake: -1.0, etc.
    - Stop counting at first INVALID guess (not incorrect, just invalid)

    Examples (up to truncation point):
    - XYYXXX salvages XYY: 0 + 0.5 + 0.25 - 0.125 = 0.625 (1 mistake, 2 correct)
    - XXXYYX salvages XXXYY: 0 + 0 + 0 + 0.125 + 0.0625 - 0.125 - 0.25 - 0.5 = -0.5625
    - Y salvages Y: 1.0 (no mistakes)
    - YXXY salvages YXXY: 1.0 + 0 + 0 + 0.125 - 0.125 - 0.25 = 0.75 (2 mistakes)
    """
    guess_history = result.get("guess_history", [])

    # Find truncation point (last correct/auto guess)
    truncation_idx = -1
    for i, guess in enumerate(guess_history):
        status = guess.get("status", "")
        if status == "invalid":
            break
        if status in ["correct", "auto"]:
            truncation_idx = i

    if truncation_idx == -1:
        # No correct guesses before invalid = nothing salvaged
        return -999.0

    # Score the salvaged portion (up to and including truncation_idx)
    score = 0.0
    mistake_count = 0

    for i in range(truncation_idx + 1):
        status = guess_history[i].get("status", "")

        if status in ["correct", "auto"]:
            # Add exponentially decayed score for correct guesses
            score += 1.0 / (2 ** i)
        elif status in ["incorrect", "one_away"]:
            # Subtract exponentially increasing penalty for mistakes
            penalty = 1.0 / (2 ** (3 - mistake_count))
            score -= penalty
            mistake_count += 1

    return score


def load_existing_results(results_dir: Path) -> List[Dict]:
    """
    Load all results_N.jsonl files from the results directory.
    Combines and deduplicates by puzzle_id, keeping the best rollout per puzzle.

    Returns:
        List of all rollout results (may have multiple per puzzle_id)
    """
    all_results = []

    # Find all results_N.jsonl files
    result_files = sorted(results_dir.glob("results_*.jsonl"))

    if not result_files:
        print(f"No existing results found in {results_dir}")
        return []

    print(f"Loading {len(result_files)} result files...")

    for result_file in result_files:
        with open(result_file, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    all_results.append(result)

    print(f"Loaded {len(all_results)} total rollouts")
    return all_results


def get_train_sft_puzzle_ids() -> Set[str]:
    """Load all puzzle IDs from the train_sft split."""
    dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
    puzzle_ids = set(example["puzzle_id"] for example in dataset)
    print(f"Loaded {len(puzzle_ids)} puzzle IDs from train_sft")
    return puzzle_ids


def select_best_rollouts(results: List[Dict]) -> Dict[str, Dict]:
    """
    Group results by puzzle_id and select the best rollout for each.

    Selection criteria for valid+won rollouts (in order):
    1. Valid guesses only (no invalid guesses) AND won the game
    2. Highest reward
    3. Shortest average assistant message tokens

    Selection criteria for non-valid/won rollouts (in order):
    1. Highest salvage quality (exponentially weighted correct guesses)
    2. Shortest average assistant message tokens (after truncation)

    Returns:
        Dict mapping puzzle_id -> best rollout
    """
    # Group by puzzle_id
    grouped = defaultdict(list)
    for result in results:
        puzzle_id = result["info"]["puzzle_id"]
        grouped[puzzle_id].append(result)

    best_rollouts = {}

    for puzzle_id, rollouts in grouped.items():
        # Sort by quality criteria
        valid_won = [r for r in rollouts if is_valid_guesses_only(r) and is_won(r)]

        if valid_won:
            # Pick best valid+won rollout
            # Sort by: highest reward, then shortest tokens
            valid_won.sort(key=lambda r: (-r["reward"], calculate_avg_assistant_tokens(r)))
            best_rollouts[puzzle_id] = valid_won[0]
        else:
            # Pick best rollout for potential doctoring
            # For each rollout, calculate salvage quality and truncate once
            rollout_data = []
            for r in rollouts:
                quality = calculate_salvage_quality(r)

                # Truncate to get token count of salvaged portion
                processed = process_rollout(r)
                original_guess_history = processed["info"].get("original_guess_history", r.get("guess_history", []))
                truncated = truncate_processed_rollout(processed, original_guess_history)

                if truncated:
                    tokens = calculate_avg_assistant_tokens(truncated)
                else:
                    tokens = calculate_avg_assistant_tokens(r)

                rollout_data.append((quality, tokens, r))

            # Sort by: highest salvage quality, then shortest tokens (after truncation)
            rollout_data.sort(key=lambda x: (-x[0], x[1]))
            best_rollouts[puzzle_id] = rollout_data[0][2]

    return best_rollouts


def prep_phase(
    best_rollouts: Dict[str, Dict],
    all_puzzle_ids: Set[str]
) -> Tuple[List[Dict], List[Dict], Set[str], Dict]:
    """
    Prep phase: Filter good/bad examples and prepare for generation.

    Returns:
        - good_examples: List of good rollouts
        - salvageable_examples: List of bad rollouts that can be doctored
        - missing_puzzle_ids: Set of puzzle IDs not in results (includes unsalvageable)
        - stats: Dict with statistics
    """
    good_examples = []
    bad_examples = []

    # Classify existing results
    for puzzle_id, rollout in best_rollouts.items():
        # Process rollout (wrap in think tags)
        processed = process_rollout(rollout)

        if is_valid_guesses_only(processed) and is_won(processed):
            good_examples.append(processed)
        else:
            bad_examples.append(processed)

    # Try to salvage bad examples through truncation
    salvageable_examples = []
    unsalvageable_puzzle_ids = set()

    for bad_example in bad_examples:
        original_guess_history = bad_example["info"].get("original_guess_history", bad_example["guess_history"])
        truncated = truncate_processed_rollout(bad_example, original_guess_history)

        if truncated and len(truncated.get("guess_history", [])) > 0:
            # Create truncated example for continuation
            salvageable = create_truncated_example(truncated)
            salvageable_examples.append(salvageable)
        else:
            # Unsalvageable - add to missing pool to start from scratch
            unsalvageable_puzzle_ids.add(bad_example["info"]["puzzle_id"])

    # Find missing puzzles (including unsalvageable ones)
    existing_puzzle_ids = set(best_rollouts.keys())
    truly_missing = all_puzzle_ids - existing_puzzle_ids
    missing_puzzle_ids = truly_missing | unsalvageable_puzzle_ids

    stats = {
        "total_puzzles": len(all_puzzle_ids),
        "good": len(good_examples),
        "bad": len(bad_examples),
        "doctoring_from_previous": len(salvageable_examples),
        "starting_from_scratch": len(missing_puzzle_ids),
        "truly_missing": len(truly_missing),
        "unsalvageable": len(unsalvageable_puzzle_ids),
    }

    return good_examples, salvageable_examples, missing_puzzle_ids, stats


def create_truncated_example(truncated_rollout: Dict) -> Dict:
    """
    Create a truncated example ready for continuation.
    Moves the truncated completion into the prompt and clears completion.

    Creates a clean dict with only the fields needed to avoid extra fields
    (like "question") that would cause format_dataset errors.
    """
    prompt = truncated_rollout.get("prompt", [])
    completion = truncated_rollout.get("completion", [])

    # Append the truncated completion to the prompt
    # This gives the model the conversation context up to the truncation point
    new_prompt = prompt + completion

    # Store the truncated guess_history in info for environment reconstruction
    truncated_guess_history = truncated_rollout.get("guess_history", [])
    info = truncated_rollout.get("info", {}).copy()
    info["resumed_from_guess_history"] = truncated_guess_history

    # Create ONLY the fields we need - don't copy everything from truncated_rollout
    # This prevents extra fields (like "question", "answer") from causing issues
    result = {
        "prompt": new_prompt,
        "completion": [],
        "info": info,
        "guess_history": [],
    }

    # Include example_id if present
    if "example_id" in truncated_rollout:
        result["example_id"] = truncated_rollout["example_id"]

    return result


async def run_evaluation_batch(
    examples: List[Dict],
    are_datasets_raw: bool,
    raw_dir: Path,
    client
) -> int:
    """Helper function to run evaluation on a batch of examples."""
    if not examples:
        return 0

    eval_dataset = Dataset.from_list(examples)

    # When are_datasets_raw=False, keep only prompt, info, and example_id columns
    # This matches what step_4 does to avoid format_dataset errors
    if not are_datasets_raw:
        columns_to_keep = ["prompt", "info"]
        if "example_id" in eval_dataset.column_names:
            columns_to_keep.append("example_id")

        columns_to_drop = [col for col in eval_dataset.column_names if col not in columns_to_keep]
        if columns_to_drop:
            eval_dataset = eval_dataset.remove_columns(columns_to_drop)

    env = ConnectionsEnv(
        ruleset="nyt",
        eval_dataset=eval_dataset,
        is_eval_dataset_raw_puzzles=are_datasets_raw
    )

    results = await env.evaluate(
        client=client,
        model="deepseek-chat",
        sampling_args={},
        num_examples=-1,
        rollouts_per_example=1,
        max_concurrent=32,
        interleave_scoring=True,
        results_path=raw_dir,
        state_columns=["guess_history", "complete_reason"],
        save_every=20,
    )

    return len(results.completion)


async def generate_phase(
    salvageable_examples: List[Dict],
    missing_puzzle_ids: Set[str],
    results_dir: Path,
    iteration: int
) -> Tuple[Path, int]:
    """
    Generate phase: Create dataset and run evaluation.

    Returns:
        - output_file: Path to the new results file
        - num_puzzles_run: Number of puzzles evaluated
    """
    # Load fresh puzzles for missing IDs
    fresh_puzzles = []
    if missing_puzzle_ids:
        full_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
        fresh_puzzles = [ex for ex in full_dataset if ex["puzzle_id"] in missing_puzzle_ids]

    if not salvageable_examples and not fresh_puzzles:
        print("No puzzles to run!")
        return None, 0

    print(f"Running evaluation on {len(salvageable_examples) + len(fresh_puzzles)} puzzles ({len(salvageable_examples)} salvaged, {len(fresh_puzzles)} fresh)")

    # Setup output
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create client once
    client = create_client()

    total_completed = 0

    try:
        # Run salvaged examples first (are_datasets_raw=False)
        if salvageable_examples:
            print(f"  Running {len(salvageable_examples)} salvaged examples...")
            completed = await run_evaluation_batch(
                salvageable_examples,
                are_datasets_raw=False,
                raw_dir=raw_dir,
                client=client
            )
            total_completed += completed
            print(f"  Completed {completed} salvaged rollouts")

        # Then run fresh puzzles (are_datasets_raw=True)
        if fresh_puzzles:
            print(f"  Running {len(fresh_puzzles)} fresh puzzles...")
            completed = await run_evaluation_batch(
                fresh_puzzles,
                are_datasets_raw=True,
                raw_dir=raw_dir,
                client=client
            )
            total_completed += completed
            print(f"  Completed {completed} fresh rollouts")

        print(f"Total completed: {total_completed} rollouts")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

    # Copy raw results to numbered results file
    raw_output = raw_dir / "results.jsonl"
    output_file = results_dir / f"results_{iteration}.jsonl"

    if raw_output.exists():
        with open(raw_output, 'r') as src, open(output_file, 'w') as dst:
            for line in src:
                dst.write(line)
        print(f"Saved results to {output_file}")
    else:
        print(f"Warning: No results file found at {raw_output}")
        return None, 0

    return output_file, len(salvageable_examples) + len(fresh_puzzles)


def print_loop_stats(loop_num: int, stats: Dict, prev_good: int = None, puzzles_run: int = None):
    """Print statistics for the current loop."""
    good = stats["good"]
    bad = stats["bad"]

    if loop_num == 1:
        # First loop: just show counts
        print(f"\nLoop {loop_num}: {good} Good, {bad} Bad")
        print(f"  - Doctoring From Previous Run: {stats['doctoring_from_previous']}")
        print(f"  - Starting from Scratch: {stats['starting_from_scratch']} ({stats['truly_missing']} new + {stats['unsalvageable']} unsalvageable)")
    else:
        # Subsequent loops: show improvement
        new_good = good - prev_good
        pct = (new_good / puzzles_run * 100) if puzzles_run > 0 else 0
        print(f"\nLoop {loop_num}: {good} Good, {bad} Bad | {new_good} ({pct:.1f}% of Ran Puzzles) -> Good Examples")
        print(f"  - Doctoring From Previous Run: {stats['doctoring_from_previous']}")
        print(f"  - Starting from Scratch: {stats['starting_from_scratch']} ({stats['truly_missing']} new + {stats['unsalvageable']} unsalvageable)")


async def main():
    parser = argparse.ArgumentParser(description="Iterative SFT example generation")
    parser.add_argument("--loop", type=int, required=True, help="Number of iterations to run")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing results files (default: generate_results/ in script directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - analyze existing results without running evaluation"
    )
    args = parser.parse_args()

    # Default to generate_results/ in the same directory as this script
    if args.results_dir is None:
        script_dir = Path(__file__).parent
        results_dir = script_dir / "generate_results"
    else:
        results_dir = Path(args.results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting iterative generation for {args.loop} loops")
    print(f"Results directory: {results_dir}")
    print("=" * 80)

    # Load train_sft puzzle IDs once
    all_puzzle_ids = get_train_sft_puzzle_ids()

    prev_good_count = None

    for loop_num in range(1, args.loop + 1):
        print(f"\n{'='*80}")
        print(f"LOOP {loop_num}/{args.loop}")
        print(f"{'='*80}")

        # 1. Load existing results
        print("\n[1/3] Loading existing results...")
        all_results = load_existing_results(results_dir)
        best_rollouts = select_best_rollouts(all_results)

        # 2. Prep phase
        print("\n[2/3] Prep phase...")
        good_examples, salvageable_examples, missing_puzzle_ids, stats = prep_phase(
            best_rollouts, all_puzzle_ids
        )

        # Print stats
        puzzles_to_run = len(salvageable_examples) + len(missing_puzzle_ids)
        print_loop_stats(loop_num, stats, prev_good_count, puzzles_to_run)

        # Check if we're done
        if stats["bad"] == 0:
            print("\n🎉 All puzzles solved! No more bad examples to improve.")
            break

        if puzzles_to_run == 0:
            print("\n⚠️  No puzzles to run (all bad examples are unsalvageable and no missing puzzles)")
            break

        # 3. Generate phase
        if args.dry_run:
            print(f"\n[3/3] Generate phase (DRY RUN - skipping actual evaluation)")
            print(f"Would run {puzzles_to_run} puzzles ({len(salvageable_examples)} salvaged + {len(missing_puzzle_ids)} fresh)")
            prev_good_count = stats["good"]
            continue

        print(f"\n[3/3] Generate phase...")
        # Determine next iteration number
        existing_files = list(results_dir.glob("results_*.jsonl"))
        next_iteration = len(existing_files) + 1

        output_file, num_run = await generate_phase(
            salvageable_examples,
            missing_puzzle_ids,
            results_dir,
            next_iteration
        )

        prev_good_count = stats["good"]

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Load and display final stats
    all_results = load_existing_results(results_dir)
    best_rollouts = select_best_rollouts(all_results)
    _, _, _, final_stats = prep_phase(best_rollouts, all_puzzle_ids)

    print(f"Total Puzzles: {final_stats['total_puzzles']}")
    print(f"Good Examples: {final_stats['good']} ({final_stats['good']/final_stats['total_puzzles']*100:.1f}%)")
    print(f"Bad Examples: {final_stats['bad']} ({final_stats['bad']/final_stats['total_puzzles']*100:.1f}%)")
    print(f"  - Doctoring From Previous Run: {final_stats['doctoring_from_previous']}")
    print(f"  - Starting from Scratch: {final_stats['starting_from_scratch']} ({final_stats['truly_missing']} new + {final_stats['unsalvageable']} unsalvageable)")


if __name__ == "__main__":
    asyncio.run(main())
