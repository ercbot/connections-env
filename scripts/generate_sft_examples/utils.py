#!/usr/bin/env python3
"""
Shared utility functions for SFT example generation pipeline.
"""

from typing import Dict, Any, List, Optional, Set


# Token limits
MAX_TOTAL_TOKENS = 2048
MAX_GENERATION_TOKENS = 1024


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
    Truncate an already-processed rollout (with think tags) to keep only messages
    up to and including the last correct guess. Also stops before any invalid guess appears.

    Returns a new dict with the truncated completion.
    """
    completion = processed_result.get("completion", [])

    # Find index of last correct guess AND first invalid guess
    last_correct_idx = -1
    first_invalid_idx = float('inf')

    for i, guess in enumerate(original_guess_history):
        if guess.get("status") == "correct":
            last_correct_idx = i
        elif guess.get("status") == "invalid":
            first_invalid_idx = min(first_invalid_idx, i)

    # Truncate at whichever comes first: before invalid guesses, or after last correct
    # If there are invalid guesses, stop before them
    if first_invalid_idx < float('inf'):
        # Find the last correct guess that came BEFORE the first invalid
        truncate_at = -1
        for i, guess in enumerate(original_guess_history):
            if i >= first_invalid_idx:
                break
            if guess.get("status") == "correct":
                truncate_at = i
    else:
        # No invalid guesses, use last correct
        truncate_at = last_correct_idx

    # Build truncated list
    # Completion alternates: assistant, user, assistant, user, ...
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

    # Truncate guess_history to match the truncation point
    truncated_guess_history = original_guess_history[:truncate_at + 1] if truncate_at >= 0 else []

    # Create new dict with truncated completion and guess_history
    truncated_result = processed_result.copy()
    truncated_result["completion"] = truncated
    truncated_result["guess_history"] = truncated_guess_history
    return truncated_result
