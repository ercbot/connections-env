#!/usr/bin/env python3
"""
Reduce tokens for all examples (good and doctored gameplay) and filter by token limits.

This script processes good_examples.jsonl and doctored_gameplay.jsonl to:
1. Check which examples exceed token limits
2. Use DeepSeek to reduce tokens in examples that exceed limits
3. Filter out examples that still exceed limits after reduction
4. Output examples within limits to their respective files

Strategy:
1. Sort assistant messages by token count (descending)
2. First pass: Reduce any messages > MAX_GENERATION_TOKENS (1024)
3. Second pass: If total > MAX_TOTAL_TOKENS, iteratively reduce longest messages
4. Target: ~80% of limits for safety margin

Args:
- good_examples_file: Path to good_examples.jsonl (default: good_examples.jsonl)
- doctored_gameplay_file: Path to doctored_gameplay.jsonl (default: doctored_gameplay.jsonl)
- --output-dir: Directory to save results (default: current directory)
- --verbose: Enable verbose output
"""

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
    get_max_total_tokens_for_puzzle,
    MAX_GENERATION_TOKENS,
)
from token_reduction_prompt import get_token_reduction_prompt


# Target percentages of limits (for safety margin)
GENERATION_TARGET_PCT = 0.80  # Target 80% of 1024 = ~819 tokens
TOTAL_TARGET_PCT = 0.80  # Target 80% of max_total (varies by puzzle size)


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


def reduce_example_tokens(
    example: Dict[str, Any],
    tokenizer,
    llm_client: OpenAI,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Reduce tokens in an example that exceeded limits.

    Returns the example with reduced token counts.
    """
    completion = example.get("completion", [])

    # Get dynamic token limit based on puzzle size
    max_total_tokens_for_puzzle = get_max_total_tokens_for_puzzle(example)

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
    if total_tokens > max_total_tokens_for_puzzle:
        target_total = int(max_total_tokens_for_puzzle * TOTAL_TARGET_PCT)
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

    # Create reduced example
    reduced_example = example.copy()
    reduced_example["completion"] = new_completion

    return reduced_example


def process_examples(
    examples: List[Dict[str, Any]],
    example_type: str,
    tokenizer,
    llm_client: OpenAI,
    verbose: bool = False
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a list of examples, reducing tokens where needed and filtering by limits.

    Returns:
        (examples_within_limits, rejection_counts)
    """
    print(f"\nProcessing {example_type} examples...")
    print(f"  Total examples: {len(examples)}")

    examples_within_limits = []
    rejection_counts = {
        "Token Limit (Total)": 0,
        "Token Limit (Generation)": 0,
    }

    for i, example in enumerate(examples):
        # Check current token metrics
        total_tokens, max_gen_tokens = calculate_token_metrics(example, tokenizer)
        max_total_tokens_for_puzzle = get_max_total_tokens_for_puzzle(example)

        # Check if reduction needed
        needs_reduction = (
            total_tokens > max_total_tokens_for_puzzle or
            max_gen_tokens > MAX_GENERATION_TOKENS
        )

        if needs_reduction:
            if verbose or (i + 1) % 10 == 0:
                print(f"  Example {i + 1}/{len(examples)}: Reducing tokens...")

            # Reduce tokens
            reduced_example = reduce_example_tokens(example, tokenizer, llm_client, verbose=verbose)

            # Check if now within limits
            new_total, new_max_gen = calculate_token_metrics(reduced_example, tokenizer)

            within_total_limit = new_total <= max_total_tokens_for_puzzle
            within_gen_limit = new_max_gen <= MAX_GENERATION_TOKENS

            if within_total_limit and within_gen_limit:
                examples_within_limits.append(reduced_example)
                if verbose:
                    print(f"    ✓ Success: {total_tokens}→{new_total} total, {max_gen_tokens}→{new_max_gen} gen")
            else:
                # Still exceeds limits after reduction
                if not within_total_limit:
                    rejection_counts["Token Limit (Total)"] += 1
                    if verbose:
                        print(f"    ✗ Failed (total): {total_tokens}→{new_total} (limit: {max_total_tokens_for_puzzle})")
                if not within_gen_limit:
                    rejection_counts["Token Limit (Generation)"] += 1
                    if verbose:
                        print(f"    ✗ Failed (gen): {max_gen_tokens}→{new_max_gen} (limit: {MAX_GENERATION_TOKENS})")
        else:
            # Already within limits
            examples_within_limits.append(example)

    print(f"  Examples within limits: {len(examples_within_limits)}/{len(examples)}")

    return examples_within_limits, rejection_counts


def print_rejection_table(rejection_counts: Dict[str, int]):
    """Print a table showing rejection counts."""
    total_rejected = sum(rejection_counts.values())

    if total_rejected == 0:
        print("\n" + "=" * 80)
        print("TOKEN LIMIT REJECTIONS")
        print("=" * 80)
        print("All examples are within token limits!")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print(f"TOKEN LIMIT REJECTIONS (N = {total_rejected})")
    print("=" * 80)
    print(f"{'Rejection Reason':<40} {'Count':<15} {'Percentage':<15}")
    print("-" * 80)

    for reason in ["Token Limit (Total)", "Token Limit (Generation)"]:
        count = rejection_counts.get(reason, 0)
        percentage = (count / total_rejected) if total_rejected > 0 else 0.0
        print(f"{reason:<40} {count:<15,} {percentage:<15.1%}")

    print("=" * 80)


def load_and_combine_doctored_examples(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Load and combine all doctored examples from doctor_results/ directory.

    Automatically finds all doctored_examples_N.jsonl files, combines them,
    and deduplicates by puzzle_id (keeping highest reward).
    """
    doctor_results_dir = output_dir / "doctor_results"

    if not doctor_results_dir.exists():
        print(f"\nWarning: doctor_results directory not found at {doctor_results_dir}")
        print("Skipping doctored examples")
        return []

    # Find all doctored_examples_N.jsonl files
    doctored_files = sorted(doctor_results_dir.glob("doctored_examples_*.jsonl"))

    if not doctored_files:
        print(f"\nWarning: No doctored_examples_*.jsonl files found in {doctor_results_dir}")
        print("Skipping doctored examples")
        return []

    print(f"\nFound {len(doctored_files)} doctoring iteration(s):")
    for f in doctored_files:
        print(f"  - {f.name}")

    # Load all examples
    all_examples = []
    for file_path in doctored_files:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))

    print(f"  Loaded {len(all_examples)} total examples")

    # Deduplicate by puzzle_id, keeping highest reward
    by_puzzle = {}
    for example in all_examples:
        puzzle_id = example.get("info", {}).get("puzzle_id")
        if not puzzle_id:
            continue

        if puzzle_id not in by_puzzle:
            by_puzzle[puzzle_id] = example
        else:
            # Keep the one with higher reward
            if example.get("reward", 0) > by_puzzle[puzzle_id].get("reward", 0):
                by_puzzle[puzzle_id] = example

    combined = list(by_puzzle.values())

    if len(all_examples) > len(combined):
        print(f"  Deduplicated to {len(combined)} unique puzzles")

    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce tokens for all examples and filter by token limits"
    )
    parser.add_argument(
        "good_examples_file",
        type=Path,
        nargs="?",
        default=Path("good_examples.jsonl"),
        help="Path to good_examples.jsonl (default: good_examples.jsonl)"
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
    good_examples_file = args.good_examples_file
    output_dir = args.output_dir
    verbose = args.verbose

    # Check for required environment variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    good_output = output_dir / "good_examples_reduced.jsonl"
    doctored_output = output_dir / "doctored_gameplay_reduced.jsonl"

    print(f"Reducing tokens for all examples...")
    print(f"  Good examples: {good_examples_file}")
    print(f"  Doctored gameplay: Automatically loading from {output_dir}/doctor_results/")
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

    # Load good examples
    good_examples = []
    if good_examples_file.exists():
        print(f"\nLoading good examples from {good_examples_file}...")
        with open(good_examples_file, "r") as f:
            for line in f:
                if line.strip():
                    good_examples.append(json.loads(line))
        print(f"  Loaded {len(good_examples)} good examples")
    else:
        print(f"\nWarning: {good_examples_file} not found, skipping good examples")

    # Load and combine doctored gameplay examples from doctor_results/
    doctored_gameplay = load_and_combine_doctored_examples(output_dir)

    # Process good examples
    good_within_limits = []
    good_rejections = {}
    if good_examples:
        good_within_limits, good_rejections = process_examples(
            good_examples,
            "good",
            tokenizer,
            llm_client,
            verbose=verbose
        )

        # Save good examples within limits
        print(f"\nSaving good examples to {good_output}...")
        with open(good_output, "w") as f:
            for example in good_within_limits:
                f.write(json.dumps(example) + "\n")
        print(f"  ✓ Saved {len(good_within_limits)} examples")

    # Process doctored gameplay examples
    doctored_within_limits = []
    doctored_rejections = {}
    if doctored_gameplay:
        doctored_within_limits, doctored_rejections = process_examples(
            doctored_gameplay,
            "doctored gameplay",
            tokenizer,
            llm_client,
            verbose=verbose
        )

        # Save doctored examples within limits
        print(f"\nSaving doctored gameplay examples to {doctored_output}...")
        with open(doctored_output, "w") as f:
            for example in doctored_within_limits:
                f.write(json.dumps(example) + "\n")
        print(f"  ✓ Saved {len(doctored_within_limits)} examples")

    # Print combined rejection statistics
    combined_rejections = {
        "Token Limit (Total)": good_rejections.get("Token Limit (Total)", 0) + doctored_rejections.get("Token Limit (Total)", 0),
        "Token Limit (Generation)": good_rejections.get("Token Limit (Generation)", 0) + doctored_rejections.get("Token Limit (Generation)", 0),
    }
    print_rejection_table(combined_rejections)

    print(f"\n✓ Token reduction complete")
    print(f"  Good examples: {len(good_within_limits)}/{len(good_examples)} within limits")
    print(f"  Doctored gameplay: {len(doctored_within_limits)}/{len(doctored_gameplay)} within limits")


if __name__ == "__main__":
    main()
