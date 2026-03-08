#!/usr/bin/env python3
"""
Token counting utilities for SFT example generation.

This module contains all token-related functionality including:
- Token counting for messages and conversations
- Token limit enforcement
- Cumulative token calculations
- Tokenizer constants and configuration
"""

from typing import Dict


# Tokenizer configuration
TOKENIZER_NAME = "ericbotti/GLM-4.6V-Flash"

# Token limits
# Note: The tokenizer only counts the last 2 thinking blocks (current + previous)
# instead of all, allowing more lenient per-round generation
MAX_GENERATION_TOKENS = 2048  # Maximum tokens per assistant message
MAX_TOTAL_TOKENS = 4096  # Maximum total tokens for entire conversation (training context limit)


def calculate_max_cumulative_tokens(result: Dict, tokenizer) -> int:
    """
    Calculate the maximum cumulative token count at any point in the conversation.

    This simulates what happens during training when the conversation is tokenized
    with a chat template. At each assistant message, we calculate the total tokens
    so far (including all previous messages).

    Args:
        result: Rollout result dict with "completion" field
        tokenizer: HuggingFace tokenizer with apply_chat_template method

    Returns:
        Maximum cumulative token count reached at any point
    """
    completion = result.get("completion", [])

    max_tokens = 0

    # Build up the conversation message by message
    for i in range(len(completion)):
        messages_so_far = completion[:i + 1]

        # Tokenize the conversation with chat template
        tokens = tokenizer.apply_chat_template(messages_so_far, tokenize=True, add_generation_prompt=False)
        cumulative_tokens = len(tokens)

        max_tokens = max(max_tokens, cumulative_tokens)

    return max_tokens


def mark_over_limit_as_invalid(result: Dict, tokenizer, token_limit: int, max_total_tokens: int) -> Dict:
    """
    Mark assistant messages as invalid if they exceed limits.

    Checks two conditions:
    1. Per-message limit: Individual message exceeds token_limit
    2. Cumulative limit: Conversation exceeds max_total_tokens

    For cumulative limit violations, compares <think> token counts between
    current and previous messages to determine which one to mark invalid.

    Args:
        result: Rollout result dict with "completion" and "guess_history"
        tokenizer: transformers tokenizer
        token_limit: Maximum tokens allowed per assistant message
        max_total_tokens: Maximum cumulative tokens for entire conversation

    Returns:
        Modified result with over-limit guesses marked as invalid
    """
    from utils import unwrap_reasoning_from_tags

    def extract_think_tokens(content: str) -> int:
        """Extract and count tokens within <think> tags."""
        reasoning_content, _ = unwrap_reasoning_from_tags(content)
        if reasoning_content:
            return len(tokenizer.encode(reasoning_content))
        return 0

    result = result.copy()
    completion = result.get("completion", [])
    guess_history = result.get("guess_history", [])

    # If no guess_history, return as-is
    if not guess_history:
        return result

    guess_history = guess_history.copy()

    # Track previous assistant message for cumulative comparison
    prev_assistant_idx = None
    prev_think_tokens = 0

    # Check each assistant message
    assistant_idx = 0
    for msg_idx, msg in enumerate(completion):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # Count tokens for full message
            msg_tokens = len(tokenizer.encode(content))

            # Count tokens for thinking portion
            think_tokens = extract_think_tokens(content)

            # Calculate cumulative tokens up to this point
            messages_so_far = completion[:msg_idx + 1]
            cumulative_token_ids = tokenizer.apply_chat_template(
                messages_so_far,
                tokenize=True,
                add_generation_prompt=False
            )
            cumulative = len(cumulative_token_ids)

            # Check 1: Per-message limit
            if msg_tokens > token_limit and assistant_idx < len(guess_history):
                guess_history[assistant_idx] = guess_history[assistant_idx].copy()
                guess_history[assistant_idx]["status"] = "invalid"
                guess_history[assistant_idx]["invalid_reason"] = f"over_token_limit_{msg_tokens}"
                # Debug: Uncomment to see per-message violations
                # print(f"  ⚠ Message {assistant_idx}: {msg_tokens} > {token_limit} tokens")

            # Check 2: Cumulative conversation limit
            if cumulative > max_total_tokens and assistant_idx < len(guess_history):
                # Compare thinking tokens: previous vs current
                if prev_assistant_idx is not None and prev_think_tokens > think_tokens:
                    # Previous message had more thinking - blame it!
                    guess_history[prev_assistant_idx] = guess_history[prev_assistant_idx].copy()
                    guess_history[prev_assistant_idx]["status"] = "invalid"
                    guess_history[prev_assistant_idx]["invalid_reason"] = f"over_cumulative_limit_prev_think_{prev_think_tokens}"
                    # Debug: Uncomment to see cumulative violations
                    # print(f"  ⚠ Cumulative {cumulative} > {max_total_tokens}: Marked msg {prev_assistant_idx} (prev_think={prev_think_tokens} > curr_think={think_tokens})")
                else:
                    # Current message has more (or equal) thinking - blame it
                    guess_history[assistant_idx] = guess_history[assistant_idx].copy()
                    guess_history[assistant_idx]["status"] = "invalid"
                    guess_history[assistant_idx]["invalid_reason"] = f"over_cumulative_limit_curr_think_{think_tokens}"
                    # Debug: Uncomment to see cumulative violations
                    # print(f"  ⚠ Cumulative {cumulative} > {max_total_tokens}: Marked msg {assistant_idx} (curr_think={think_tokens} >= prev_think={prev_think_tokens})")

            # Track for next iteration
            prev_assistant_idx = assistant_idx
            prev_think_tokens = think_tokens
            assistant_idx += 1

    result["guess_history"] = guess_history
    return result


def analyze_overlimit_messages(result: Dict, tokenizer, token_limit: int) -> Dict:
    """
    Analyze a rollout to find assistant messages over the token limit.

    This is used for diagnostic purposes and by the interactive editing tool
    to identify which messages need manual trimming.

    Args:
        result: Rollout result dict (should be processed with think tags)
        tokenizer: HuggingFace tokenizer
        token_limit: Maximum tokens allowed per assistant message

    Returns:
        Dict with:
        - num_over: Number of messages over limit
        - total_tokens_over: Sum of excess tokens across all messages
        - max_excess: Maximum excess tokens in any single message
        - overlimit_details: List of dicts with per-message details
    """
    completion = result.get("completion", [])

    overlimit_details = []
    total_tokens_over = 0

    assistant_idx = 0
    for msg_idx, msg in enumerate(completion):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            tokens = len(tokenizer.encode(content))

            if tokens > token_limit:
                excess = tokens - token_limit
                overlimit_details.append({
                    "message_idx": assistant_idx,
                    "completion_idx": msg_idx,  # Index in the full completion array
                    "tokens": tokens,
                    "excess": excess,
                    "content": content
                })
                total_tokens_over += excess

            assistant_idx += 1

    max_excess = max((d["excess"] for d in overlimit_details), default=0)

    return {
        "num_over": len(overlimit_details),
        "total_tokens_over": total_tokens_over,
        "overlimit_details": overlimit_details,
        "max_excess": max_excess
    }
