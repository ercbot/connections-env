#!/usr/bin/env python3
"""
Shared utility functions for SFT example generation pipeline.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from openai import AsyncOpenAI


# Token limits
MAX_TOTAL_TOKENS = 2048
MAX_GENERATION_TOKENS = 1024


def get_num_words_in_puzzle(result: Dict[str, Any]) -> int:
    """Get the total number of words in the puzzle."""
    categories = result.get("info", {}).get("categories", [])
    total_words = sum(len(cat.get("members", [])) for cat in categories)
    return total_words


def get_max_total_tokens_for_puzzle(result: Dict[str, Any]) -> int:
    """Get the max total tokens limit based on puzzle size."""
    num_words = get_num_words_in_puzzle(result)
    if num_words <= 16:
        return MAX_TOTAL_TOKENS  # 2048 for standard puzzles
    else:
        # Dynamic limit for larger puzzles, capped at 3072
        return min(3072, 128 * num_words)


def is_valid_guesses_only(result: Dict[str, Any]) -> bool:
    """Check if all guesses were valid (no invalid guesses)."""
    guess_history = result.get("guess_history", [])
    return not any(guess.get("status") == "invalid" for guess in guess_history)


def is_won(result: Dict[str, Any]) -> bool:
    """Check if the game was won."""
    winning_reasons = ["all_categories_found", "theme_guessing_complete"]
    return result.get("complete_reason") in winning_reasons


def calculate_token_metrics(result: Dict[str, Any], tokenizer) -> tuple[int, int]:
    """
    Calculate token metrics for a rollout.

    Returns:
        (total_tokens, max_generation_tokens)
        - total_tokens: Total tokens excluding last message if not assistant
        - max_generation_tokens: Maximum tokens in any single assistant message
    """
    completion = result.get("completion", [])

    # Calculate total tokens (excluding last message if not assistant)
    messages_for_total = completion
    if messages_for_total and messages_for_total[-1].get("role") != "assistant":
        messages_for_total = messages_for_total[:-1]

    total_tokens = 0
    max_generation_tokens = 0

    for msg in messages_for_total:
        msg_tokens = len(tokenizer.encode(msg["content"]))
        total_tokens += msg_tokens

        # Track max for assistant messages
        if msg.get("role") == "assistant":
            max_generation_tokens = max(max_generation_tokens, msg_tokens)

    return total_tokens, max_generation_tokens


def wrap_reasoning_in_tags(content: str) -> str:
    """
    Wrap all parts of the assistant message before <guess> tags in <think> tags.

    If <think> tags are already present, returns content unchanged to avoid double-wrapping.

    Example:
    "Let's think. <guess>[`WORD1`]</guess>"
    -> "<think>Let's think.</think>\n\n<guess>[`WORD1`]</guess>"
    """
    # Check if think tags are already present
    if "<think>" in content:
        return content

    if "<guess>" not in content:
        # No guess tag, wrap entire content
        return f"<think>{content}</think>"

    # Split on <guess> tag
    parts = content.split("<guess>", 1)
    reasoning = parts[0]
    guess_and_rest = parts[1]

    # Strip trailing whitespace from reasoning and wrap in think tags
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


def get_found_category_indices(guess_history: List[Dict[str, Any]]) -> Set[int]:
    """Get the indices of categories that have been found."""
    found_indices = set()
    for guess in guess_history:
        if guess.get("status") == "correct":
            category_idx = guess.get("category_idx")
            if category_idx is not None:
                found_indices.add(int(category_idx))
    return found_indices


def get_guessed_categories_in_order(
    guess_history: List[Dict[str, Any]],
    categories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get categories that were correctly guessed, in the order they were guessed.

    Returns a list of category dicts in the order they were found.
    """
    guessed_categories = []
    for guess in guess_history:
        if guess.get("status") == "correct":
            category_idx = guess.get("category_idx")
            if category_idx is not None and category_idx < len(categories):
                guessed_categories.append(categories[int(category_idx)])
    return guessed_categories


def truncate_processed_rollout(
    processed_result: Dict[str, Any],
    original_guess_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Truncate an already-processed rollout (with think tags) to prepare it for doctoring.

    Truncation logic:
    1. Remove all invalid guesses (and everything after the first invalid)
    2. Check if the last remaining guess would cause the game to be lost
       (i.e., if it's an incorrect guess that puts us at max_mistakes)
    3. If so, remove that losing guess too
    4. Keep all other guesses (including incorrect ones that don't cause a loss)

    This ensures the doctored example can be completed successfully.

    Returns a new dict with the truncated completion.
    """
    completion = processed_result.get("completion", [])
    info = processed_result.get("info", {})

    # Get ruleset configuration to determine max_mistakes
    # Default to NYT rules if not specified
    from connections.rulesets import get_ruleset_config
    ruleset_config = get_ruleset_config("nyt")

    # Step 1: Find first invalid guess and truncate before it
    first_invalid_idx = float('inf')
    for i, guess in enumerate(original_guess_history):
        if guess.get("status") == "invalid":
            first_invalid_idx = min(first_invalid_idx, i)

    # Truncate guess history at first invalid (or keep all if no invalid)
    if first_invalid_idx < float('inf'):
        truncated_guess_history = original_guess_history[:first_invalid_idx]
    else:
        truncated_guess_history = original_guess_history.copy()

    # Step 2: Simulate game state to check if last guess causes loss
    # We need to count mistakes according to the ruleset rules
    mistakes = 0
    found_categories = 0
    categories = info.get("categories", [])
    total_categories = len(categories)

    for i, guess in enumerate(truncated_guess_history):
        status = guess.get("status")

        if status == "correct":
            found_categories += 1
        elif status in ["incorrect", "one_away"]:
            # Check if we should count this mistake based on ruleset
            remaining_categories = total_categories - found_categories
            threshold = ruleset_config.mistakes_count_when_x_categories_remain

            should_count = (threshold == "any") or (remaining_categories <= threshold)

            if should_count:
                mistakes += 1

                # Check if this guess causes us to hit max_mistakes (game loss)
                if mistakes >= ruleset_config.max_mistakes:
                    # This is the losing guess - truncate before it
                    truncated_guess_history = truncated_guess_history[:i]
                    break

    # Step 3: Build truncated completion messages based on truncated_guess_history
    truncate_at = len(truncated_guess_history) - 1  # Index of last guess to keep

    truncated = []
    assistant_count = 0

    for msg in completion:
        if msg["role"] == "assistant":
            if assistant_count <= truncate_at:
                truncated.append(msg)
                assistant_count += 1
            else:
                break  # Stop including messages after truncation point
        elif msg["role"] == "user":
            # Include user feedback only if we haven't passed the truncation point
            if assistant_count <= truncate_at + 1:
                truncated.append(msg)

    # Create new dict with truncated completion and guess_history
    truncated_result = processed_result.copy()
    truncated_result["completion"] = truncated
    truncated_result["guess_history"] = truncated_guess_history
    return truncated_result


def setup_output_directories(
    base_output_dir: Path,
    results_subdir: str = "",
    filename_pattern: str = "results_{}.jsonl"
) -> Tuple[Path, Path, Path, int, Path]:
    """
    Set up output directory structure and find next iteration number.
    
    Creates:
    - base_output_dir/
    - base_output_dir/raw/ (for raw results from evaluate())
    - base_output_dir/results_subdir/ (optional subdirectory for final results)
    
    Args:
        base_output_dir: Base output directory
        results_subdir: Optional subdirectory name for final results (e.g., "doctor_results")
        filename_pattern: Pattern for numbered output files, with {} placeholder for iteration
        
    Returns:
        Tuple of (base_output_dir, raw_results_dir, final_output_dir, iteration, output_file)
        - base_output_dir: The base output directory
        - raw_results_dir: Directory for raw results (base_output_dir/raw)
        - final_output_dir: Directory for final results (base_output_dir/results_subdir or base_output_dir)
        - iteration: Next available iteration number
        - output_file: Path to the numbered output file
    """
    base_output_dir = base_output_dir.resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create raw results subdirectory
    raw_results_dir = base_output_dir / "raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine final output directory
    if results_subdir:
        final_output_dir = base_output_dir / results_subdir
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        final_output_dir = base_output_dir
    
    # Find next iteration number
    iteration = 1
    while (final_output_dir / filename_pattern.format(iteration)).exists():
        iteration += 1
    
    output_file = final_output_dir / filename_pattern.format(iteration)
    
    return base_output_dir, raw_results_dir, final_output_dir, iteration, output_file


def copy_raw_results_to_output(
    raw_results_dir: Path,
    output_file: Path,
    raw_filename: str = "results.jsonl"
) -> bool:
    """
    Copy raw results file to numbered output file.
    
    Args:
        raw_results_dir: Directory containing raw results
        output_file: Destination file path
        raw_filename: Name of the raw results file (default: "results.jsonl")
        
    Returns:
        True if copy was successful, False if raw file doesn't exist
    """
    raw_results_file = raw_results_dir / raw_filename
    if raw_results_file.exists():
        shutil.copy2(raw_results_file, output_file)
        return True
    return False


def create_client():
    """Get a client for the LLM."""
    return AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )