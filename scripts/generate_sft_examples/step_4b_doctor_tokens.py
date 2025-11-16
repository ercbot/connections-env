#!/usr/bin/env python3
"""
Doctor token-limited examples by reducing verbose reasoning.

This script loads doctor_tokens.jsonl (examples that won with valid guesses
but exceeded token limits) and uses DeepSeek to intelligently reduce the token
count while preserving reasoning quality.

Strategy:
1. Sort assistant messages by token count (descending)
2. First pass: Reduce any messages > MAX_GENERATION_TOKENS (1024)
3. Second pass: If total > MAX_TOTAL_TOKENS (2048), iteratively reduce longest messages
4. Target: ~80% of limits for safety margin
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer
from openai import OpenAI

# Import shared utilities
from utils import (
    calculate_token_metrics,
    MAX_TOTAL_TOKENS,
    MAX_GENERATION_TOKENS,
)
from token_reduction_prompt import get_token_reduction_prompt


# Target percentages of limits (for safety margin)
GENERATION_TARGET_PCT = 0.80  # Target 80% of 1024 = ~819 tokens
TOTAL_TARGET_PCT = 0.80  # Target 80% of 2048 = ~1638 tokens


def extract_guess_from_message(content: str) -> str | None:
    """Extract the <guess> tag content from a message."""
    if "<guess>" not in content:
        return None

    start = content.find("<guess>")
    end = content.find("</guess>", start)
    if end == -1:
        return None

    return content[start:end + len("</guess>")]


def extract_think_content(content: str) -> str | None:
    """Extract content from <think> tags."""
    if "<think>" not in content:
        return None

    start = content.find("<think>")
    end = content.find("</think>", start)
    if end == -1:
        return None

    # Return just the content inside the tags (not including the tags themselves)
    return content[start + len("<think>"):end]


def verify_single_think_tag(content: str) -> bool:
    """Verify there is exactly one set of <think></think> tags."""
    return content.count("<think>") == 1 and content.count("</think>") == 1


def replace_think_content(original: str, new_think_content: str) -> str:
    """Replace the content inside <think> tags in the original message."""
    if "<think>" not in original:
        return original

    start = original.find("<think>")
    end = original.find("</think>", start)
    if end == -1:
        return original

    # Reconstruct with new think content
    before_think = original[:start + len("<think>")]
    after_think = original[end:]

    return before_think + new_think_content + after_think


def verify_guess_unchanged(original: str, reduced: str) -> bool:
    """Verify that the guess in the reduced message matches the original."""
    original_guess = extract_guess_from_message(original)
    reduced_guess = extract_guess_from_message(reduced)

    return original_guess == reduced_guess


def reduce_message_tokens(
    message_content: str,
    current_tokens: int,
    target_tokens: int,
    tokenizer,
    llm_client: OpenAI,
    is_generation_limit: bool = False,
    verbose: bool = False
) -> str:
    """
    Use LLM to reduce tokens in a single assistant message.

    Args:
        message_content: Original message content
        current_tokens: Current token count
        target_tokens: Target token count
        tokenizer: Tokenizer for counting tokens
        llm_client: OpenAI client for LLM calls
        is_generation_limit: True if reducing for generation limit
        verbose: Enable verbose logging

    Returns:
        Reduced message content
    """
    if verbose:
        print(f"    Reducing message from {current_tokens} to ~{target_tokens} tokens...")

    # Get reduction prompt
    prompt = get_token_reduction_prompt(
        message_content,
        current_tokens,
        target_tokens,
        is_generation_limit
    )

    # Call LLM
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        llm_response = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if llm_response.startswith("```"):
            lines = llm_response.split("\n")
            llm_response = "\n".join(lines[1:-1]) if len(lines) > 2 else llm_response

        # Verify LLM response has exactly one set of <think> tags
        if not verify_single_think_tag(llm_response):
            if verbose:
                print(f"    WARNING: LLM response has invalid <think> tag structure, using original")
            return message_content

        # Extract the new think content from LLM response
        new_think_content = extract_think_content(llm_response)
        if new_think_content is None:
            if verbose:
                print(f"    WARNING: Could not extract <think> content from LLM response, using original")
            return message_content

        # Replace think content in original message (preserves everything else)
        reduced_content = replace_think_content(message_content, new_think_content)

        # Verify guess unchanged
        if not verify_guess_unchanged(message_content, reduced_content):
            if verbose:
                print(f"    WARNING: Guess changed during reduction, using original")
            return message_content

        # Check actual token count
        actual_tokens = len(tokenizer.encode(reduced_content))
        if verbose:
            print(f"    Result: {actual_tokens} tokens (target was {target_tokens})")

        return reduced_content

    except Exception as e:
        print(f"    ERROR during reduction: {e}")
        if verbose:
            print(f"    Using original message")
        return message_content


def doctor_token_limited_example(
    example: Dict[str, Any],
    tokenizer,
    llm_client: OpenAI,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Reduce tokens in an example that exceeded limits.

    Returns the doctored example with reduced token counts.
    """
    completion = example.get("completion", [])

    # Get assistant messages with their indices and token counts
    assistant_messages = []
    for i, msg in enumerate(completion):
        if msg.get("role") == "assistant":
            tokens = len(tokenizer.encode(msg["content"]))
            assistant_messages.append({
                "index": i,
                "content": msg["content"],
                "tokens": tokens,
            })

    if not assistant_messages:
        return example

    # Sort by token count (descending) to reduce longest first
    assistant_messages.sort(key=lambda x: x["tokens"], reverse=True)

    # Create a mutable copy of completion
    new_completion = [msg.copy() for msg in completion]

    # Phase 1: Reduce any messages exceeding generation limit
    phase1_reductions = 0
    for msg_data in assistant_messages:
        if msg_data["tokens"] > MAX_GENERATION_TOKENS:
            target = int(MAX_GENERATION_TOKENS * GENERATION_TARGET_PCT)
            phase1_reductions += 1

            reduced_content = reduce_message_tokens(
                msg_data["content"],
                msg_data["tokens"],
                target,
                tokenizer,
                llm_client,
                is_generation_limit=True,
                verbose=verbose
            )

            # Update in new_completion
            new_completion[msg_data["index"]]["content"] = reduced_content

            # Update token count for next phase
            msg_data["content"] = reduced_content
            msg_data["tokens"] = len(tokenizer.encode(reduced_content))

    # Phase 2: If total still exceeds limit, reduce further
    # Calculate current total
    total_tokens, _ = calculate_token_metrics({"completion": new_completion}, tokenizer)

    phase2_reductions = 0
    if total_tokens > MAX_TOTAL_TOKENS:
        target_total = int(MAX_TOTAL_TOKENS * TOTAL_TARGET_PCT)
        reduction_needed = total_tokens - target_total

        # Sort again by current token counts (may have changed in phase 1)
        assistant_messages.sort(key=lambda x: x["tokens"], reverse=True)

        # Iteratively reduce longest messages
        for msg_data in assistant_messages:
            if reduction_needed <= 0:
                break

            # Reduce this message proportionally
            reduction_for_this_msg = min(
                reduction_needed,
                int(msg_data["tokens"] * 0.3)  # Max 30% reduction per message
            )

            if reduction_for_this_msg > 0:
                target = msg_data["tokens"] - reduction_for_this_msg
                phase2_reductions += 1

                reduced_content = reduce_message_tokens(
                    msg_data["content"],
                    msg_data["tokens"],
                    target,
                    tokenizer,
                    llm_client,
                    is_generation_limit=False,
                    verbose=verbose
                )

                # Update in new_completion
                new_completion[msg_data["index"]]["content"] = reduced_content

                # Update tracking
                new_tokens = len(tokenizer.encode(reduced_content))
                actual_reduction = msg_data["tokens"] - new_tokens
                reduction_needed -= actual_reduction
                msg_data["tokens"] = new_tokens

    # Create doctored example
    doctored = example.copy()
    doctored["completion"] = new_completion

    return doctored


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Doctor token-limited examples by reducing verbose reasoning"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("doctor_tokens.jsonl"),
        help="Path to doctor_tokens.jsonl (default: doctor_tokens.jsonl in current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save results (default: current directory)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    verbose = args.verbose

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    # Check for required environment variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "doctored_tokens.jsonl"

    print(f"Doctoring token-limited examples...")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Model: deepseek-chat")
    print()

    # Load tokenizer
    print("Loading tokenizer (PrimeIntellect/Qwen3-4b)...")
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-4b")

    # Initialize DeepSeek client (uses OpenAI-compatible API)
    llm_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # Load token-limited examples
    print(f"Loading examples from {input_file}...")
    token_examples = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                token_examples.append(json.loads(line))

    print(f"  Loaded {len(token_examples)} token-limited examples")
    print()

    # Doctor each example
    print("Processing examples...")
    doctored_examples = []
    success_count = 0

    for i, example in enumerate(token_examples):
        # Check current token metrics
        total_tokens, max_gen_tokens = calculate_token_metrics(example, tokenizer)

        # Doctor the example
        doctored = doctor_token_limited_example(example, tokenizer, llm_client, verbose=verbose)
        doctored_examples.append(doctored)

        # Verify results
        new_total, new_max_gen = calculate_token_metrics(doctored, tokenizer)

        within_limits = new_total <= MAX_TOTAL_TOKENS and new_max_gen <= MAX_GENERATION_TOKENS
        if within_limits:
            success_count += 1
            status = "✓"
        else:
            status = "✗"

        # Concise progress output
        print(f"{status} {i + 1}/{len(token_examples)}: {total_tokens}→{new_total} total, {max_gen_tokens}→{new_max_gen} gen")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        for example in doctored_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n✓ Successfully doctored {len(doctored_examples)} examples ({success_count} within limits)")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
