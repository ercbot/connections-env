#!/usr/bin/env python3
"""
Shared utility functions for SFT example generation pipeline.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from openai import AsyncOpenAI

def is_valid_guesses_only(result: Dict[str, Any]) -> bool:
    """Check if all guesses were valid (no invalid guesses)."""
    guess_history = result.get("guess_history", [])
    if guess_history is None:
        return False  # Treat None as invalid
    return not any(guess.get("status") == "invalid" for guess in guess_history)


def is_won(result: Dict[str, Any]) -> bool:
    """Check if the game was won."""
    winning_reasons = ["all_categories_found", "theme_guessing_complete"]
    return result.get("complete_reason") in winning_reasons


def is_structurally_valid(result: Dict[str, Any]) -> bool:
    """
    Check that assistant message count matches non-auto guess count.

    For resumed/doctored rollouts, earlier assistant messages may live in
    the prompt, so we count across both prompt and completion.
    """
    prompt = result.get("prompt", [])
    completion = result.get("completion", [])
    guess_history = result.get("guess_history", [])

    all_assistant_msgs = [
        m for m in prompt + completion if m.get("role") == "assistant"
    ]
    non_auto_guesses = [
        g for g in guess_history if g.get("status") != "auto"
    ]
    return len(all_assistant_msgs) == len(non_auto_guesses)


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


def normalize_prompt_completion(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a rollout so that only system + initial user message stay in the
    prompt, and all subsequent conversation messages are in the completion.

    For doctored/resumed rollouts, earlier assistant+user turns get moved into
    the prompt by create_truncated_example. This causes indexing mismatches in
    downstream functions (mark_over_limit_as_invalid, truncate_processed_rollout)
    that assume completion assistant messages align 1:1 with guess_history entries.

    Normalizing here ensures a consistent layout regardless of how many times
    a rollout has been doctored.
    """
    prompt = result.get("prompt", [])
    completion = result.get("completion", [])

    # Find the split point: keep system + first user message in prompt
    split = 0
    for i, msg in enumerate(prompt):
        split = i + 1
        if msg.get("role") == "user":
            break

    normalized = result.copy()
    normalized["prompt"] = prompt[:split]
    normalized["completion"] = prompt[split:] + completion
    # Deep copy info so downstream mutations don't leak back to the original
    if "info" in normalized:
        normalized["info"] = normalized["info"].copy()
    return normalized


def process_rollout(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a rollout by normalizing prompt/completion split and wrapping
    assistant reasoning in <think> tags.

    Returns a new dict with the processed completion messages.
    """
    processed = normalize_prompt_completion(result)
    completion = processed.get("completion", [])

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
    2. Remove all trailing incorrect/one_away guesses after the last correct guess
    
    This ensures we resume from the last point where the model made progress.

    Returns a new dict with the truncated completion.
    """
    completion = processed_result.get("completion", [])

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

    # Step 2: Find the last correct guess and remove all trailing incorrect/one_away guesses
    last_correct_idx = -1
    for i, guess in enumerate(truncated_guess_history):
        if guess.get("status") == "correct":
            last_correct_idx = i

    # If we found a correct guess, truncate after it (removing trailing failures)
    if last_correct_idx >= 0:
        truncated_guess_history = truncated_guess_history[:last_correct_idx + 1]

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


MODEL_PROVIDERS = {
    "deepseek-chat": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "deepseek-reasoner": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
}

# Models not in MODEL_PROVIDERS are assumed to be OpenAI models
OPENAI_DEFAULT = {
    "api_key_env": "OPENAI_API_KEY",
}


def create_client(model: str) -> AsyncOpenAI:
    """Get a client for the given model."""
    if model in MODEL_PROVIDERS:
        provider = MODEL_PROVIDERS[model]
        return AsyncOpenAI(
            api_key=os.getenv(provider["api_key_env"]),
            base_url=provider["base_url"],
        )
    # Default to OpenAI
    return AsyncOpenAI(
        api_key=os.getenv(OPENAI_DEFAULT["api_key_env"]),
    )