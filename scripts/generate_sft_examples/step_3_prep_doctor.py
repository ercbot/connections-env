#!/usr/bin/env python3
"""
Prepare bad examples for gameplay doctoring

Args:
- input_file: Path to the bad_examples.jsonl file

For each bad example with gameplay issues:
1. Only process examples with rejection_reason "Invalid Guess" or "Game Lost"

2. Truncate the example to keep only messages up to and including the last correct guess.
   - Also stops before any invalid guess appears.

3. Create a doctoring-ready example with a system prompt that instructs which categories to guess:
   - Preserve the order of categories that were already correctly guessed
   - Rank remaining categories by difficulty using llm
   - Create system prompt with instructions for ALL categories in the optimized order

4. Save to doctor_gameplay.jsonl

NOTE: Token reduction is now handled in step_5_reduce_tokens.py
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

# Import shared utilities
from utils import (
    get_found_category_indices,
    get_guessed_categories_in_order,
    truncate_processed_rollout,
)


# Rejection reasons that indicate gameplay issues (only these are handled in this step)
GAMEPLAY_REJECTION_REASONS = {"Invalid Guess", "Game Lost"}


class CategoryRanking(BaseModel):
    """Ranking of categories from easiest to hardest (1-indexed)."""
    ranking: List[int] = Field(description="List of category numbers (1-indexed) in order from easiest to hardest")


def rank_categories_by_difficulty(
    categories: List[Dict[str, Any]],
    llm_client: OpenAI,
    model: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """
    Have LLM rank categories by perceived difficulty to guess.
    
    Returns categories sorted from easiest to hardest.
    """
    # If there's only one category, return it as-is
    if len(categories) == 1:
        return categories
    
    # Build prompt for ranking
    categories_text = "\n\n".join([
        f"Category {i+1}:\n"
        f"Theme: {cat['group']}\n"
        f"Words: {', '.join([f'`{w}`' for w in cat['members']])}"
        for i, cat in enumerate(categories)
    ])
    
    ranking_prompt = f"""You are helping rank Connections puzzle categories by difficulty.

Below are {len(categories)} categories from a Connections puzzle. Your task is to rank them from EASIEST to HARDEST based on how easy it would be for a player to identify the pattern and group the items correctly.

Consider:
- How obvious is the pattern/theme?
- How many potential false connections exist?
- How specialized is the knowledge required?
- How tricky or misleading are the items?

{categories_text}

Respond with ONLY a JSON array of category numbers in order from easiest to hardest.
Example: [3, 1, 4, 2]

Your ranking (easiest to hardest):"""
    
    # Call LLM with structured output
    try:
        response = llm_client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": ranking_prompt}],
            temperature=0.3,
            response_format=CategoryRanking,
        )
        
        ranking_result = response.choices[0].message.parsed
        ranking = ranking_result.ranking
        
        # Validate ranking
        if len(ranking) != len(categories):
            print(f"Warning: Ranking length ({len(ranking)}) doesn't match categories ({len(categories)}), using original order")
            return categories
        
        # Check if all indices are valid (1-indexed)
        if not all(1 <= r <= len(categories) for r in ranking):
            print(f"Warning: Invalid category numbers in ranking, using original order")
            return categories
        
        # Convert to 0-indexed and reorder categories
        ranking = [r - 1 for r in ranking]
        return [categories[i] for i in ranking]
    except Exception as e:
        print(f"Warning: Failed to get ranking from LLM, using original order: {e}")
        return categories


def create_doctoring_system_prompt(
    base_system_prompt: str,
    all_categories: List[Dict[str, Any]],
    truncated_guess_history: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    ruleset_config
) -> str:
    """
    Create a system prompt with explicit instructions for which categories to guess in each round.

    Args:
        base_system_prompt: The base system prompt from the environment
        all_categories: List of ALL categories to guess going forward (in order)
        truncated_guess_history: Full guess history from truncation (includes correct AND incorrect)
        categories: All categories in the puzzle (for looking up themes)
        ruleset_config: Ruleset configuration

    Returns:
        Modified system prompt with doctoring instructions
    """
    if not all_categories:
        return base_system_prompt

    # Build unified category list - all guesses in sequential order
    category_list = []
    category_list.append("**Make the following guesses in exact order**")

    # Add all guesses from history (correct and incorrect)
    for i, guess_record in enumerate(truncated_guess_history, start=1):
        # Handle both old "words" field and new "items" field for backward compatibility
        items = guess_record.get("items", guess_record.get("words", []))
        status = guess_record.get("status")
        category_idx = guess_record.get("category_idx")

        guess_str = ", ".join(f"`{item}`" for item in items)

        # Determine label based on status
        if status == "correct" and category_idx is not None and category_idx < len(categories):
            # Correct guess - show the theme
            theme = categories[int(category_idx)].get("group", "Unknown")
            label = f"Theme: {theme}"
        else:
            # Incorrect or one_away - show as Red Herring
            label = "Red Herring"

        category_list.append(f"{i}. <guess>[{guess_str}]</guess>\n   ({label})")

    # Add remaining categories to guess
    num_already_guessed = len(truncated_guess_history)
    for i, category in enumerate(all_categories, start=num_already_guessed + 1):
        theme = category.get("group", "Unknown")
        members = category.get("members", [])
        guess_str = ", ".join(f"`{item}`" for item in members)
        category_list.append(f"{i}. <guess>[{guess_str}]</guess>\n   (Theme: {theme})")

    categories_text = "\n".join(category_list)
    
    # Build the doctoring instructions as an f-string literal
    doctoring_instructions = f"""
---
## SPECIAL TRAINING DATA GENERATION MODE

You are engaging in a special version of the Connections game environment described above. This special mode is designed to create high-quality training data for supervised fine-tuning (SFT).

### Instructions
- The items to guess for each round are provided to you below in a unified list.
- Make guesses in the exact order specified in the list below.
- Some guesses may be marked as "Red Herring" (incorrect guesses) - this is intentional and simulates how even correct reasoning can lead to incorrect guesses in these puzzles.
- You must generate natural, logical reasoning that would lead you to the conclusion that the items belong to the category.
- Your reasoning should feel authentic and demonstrate genuine problem-solving thought processes.
- Do NOT mention any meta-knowledge like "I was told to guess X" or "The prompt indicates the next guess should be Y"
- Your thinking should not directly arrive at the conclusion you have been informed of - first consider potential red herrings, alternative groupings, etc.

### Example Guess
<think>
Looking at the remaining items, I can see several potential groupings. At first, I considered whether `ITEM1`, `ITEM5`, and `ITEM7` might go together because they all seem related to movement, but I'm not confident about a fourth item that would complete that category.

I also notice that `ITEM2` and `ITEM6` could potentially be part of an item pattern or sound-alike category, but again I'm struggling to find the other two members with certainty.

However, when I look more carefully at `ITEM1`, `ITEM2`, `ITEM3`, and `ITEM4`, I notice they share a stronger connection - they all relate to [insert specific reasoning about the actual theme]. This feels like the most cohesive group, even though some of these items could arguably fit other patterns. The connection is clear enough that I'm confident this is the intended category.
</think>

<guess>[`ITEM1`, `ITEM2`, `ITEM3`, `ITEM4`]</guess>

### Item Guessing Plan:
{categories_text}

### Remember:
- Make guesses in the exact order specified above.
- Make exactly one guess per round following the sequential order.
- Generate natural, logical reasoning for each guess.
- Use the required <think>...</think> and <guess>...</guess> tag format.
- Do NOT reference this meta-instruction or the fact that answers are provided.
- Act as if you are naturally discovering the connections through reasoning.
- Your responses should read as if you are genuinely solving the puzzle.
"""
    
    return base_system_prompt + doctoring_instructions


def create_doctor_ready_example(
    truncated_result: Dict[str, Any],
    original_guess_history: List[Dict[str, Any]],
    ruleset_config,
    llm_client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Create a doctoring-ready example with a system prompt that instructs which categories to guess.

    This replaces the system prompt with one that has explicit instructions for remaining categories.
    The order is determined by looking at the ORIGINAL (pre-truncation) guess history to preserve
    the model's natural guessing preferences.

    Args:
        truncated_result: The truncated rollout result
        original_guess_history: The ORIGINAL guess history before truncation
        ruleset_config: Ruleset configuration
        llm_client: Optional LLM client for ranking (fallback if original history doesn't provide order)

    Strategy:
    1. Use truncated guess history to determine what was already guessed
    2. Look at ORIGINAL guess history to see what the model tried to guess next (even if invalid)
    3. Order remaining categories based on original attempt order
    4. Only use LLM ranking as fallback if original history doesn't cover all categories
    """
    # Get categories and truncated guess history
    categories = truncated_result.get("info", {}).get("categories", [])
    truncated_guess_history = truncated_result.get("guess_history", [])

    # Determine which categories were already found (from truncated history)
    found_indices = get_found_category_indices(truncated_guess_history)

    # Extract category ordering from ORIGINAL guess history (including invalid/incorrect guesses)
    # This preserves the model's natural guessing preference
    attempted_category_order = []
    seen_categories = set()

    for guess in original_guess_history:
        category_idx = guess.get("category_idx")
        status = guess.get("status")

        # For correct guesses, we know the category
        if status == "correct" and category_idx is not None:
            cat_idx = int(category_idx)
            if cat_idx not in seen_categories and cat_idx not in found_indices:
                attempted_category_order.append(categories[cat_idx])
                seen_categories.add(cat_idx)

        # For incorrect/one_away guesses with a category_idx, they were trying for that category
        elif status in ["incorrect", "one_away"] and category_idx is not None:
            cat_idx = int(category_idx)
            if cat_idx not in seen_categories and cat_idx not in found_indices:
                attempted_category_order.append(categories[cat_idx])
                seen_categories.add(cat_idx)

    # Get any remaining categories not covered by original attempts
    remaining_categories = [
        cat for idx, cat in enumerate(categories)
        if idx not in found_indices and idx not in seen_categories
    ]

    # Rank the remaining categories (not covered by original attempts) using LLM
    if remaining_categories and llm_client:
        try:
            ranked_remaining = rank_categories_by_difficulty(remaining_categories, llm_client)
        except Exception as e:
            print(f"Warning: Failed to rank categories, using original order: {e}")
            ranked_remaining = remaining_categories
    else:
        ranked_remaining = remaining_categories

    # Combine: attempted order (from original history) + LLM-ranked remaining
    ranked_remaining = attempted_category_order + ranked_remaining

    # If no remaining categories to guess, return as-is (shouldn't happen for doctoring)
    if not ranked_remaining:
        return truncated_result
    
    # Get the original prompt and completion (truncated)
    prompt = truncated_result.get("prompt", [])
    completion = truncated_result.get("completion", [])

    # Find and replace the system prompt in the original prompt
    new_prompt = []
    base_system_prompt = ""

    # Get the truncated guess history for the system prompt
    truncated_guess_history = truncated_result.get("guess_history", [])

    for msg in prompt:
        if msg.get("role") == "system":
            base_system_prompt = msg.get("content", "")
            # Create new system prompt with doctoring instructions
            # Pass the full guess history (correct AND incorrect) and remaining categories
            new_system_prompt = create_doctoring_system_prompt(
                base_system_prompt,
                ranked_remaining,  # Only remaining categories to guess
                truncated_guess_history,  # ALL guesses made so far
                categories,  # All categories for theme lookup
                ruleset_config
            )
            new_prompt.append({
                "role": "system",
                "content": new_system_prompt
            })
        else:
            new_prompt.append(msg)

    # Append the truncated completion to the prompt
    # This gives the model the conversation context up to the truncation point
    new_prompt.extend(completion)

    # Create doctoring-ready example
    doctor_ready = truncated_result.copy()
    doctor_ready["prompt"] = new_prompt
    # Clear the completion - we'll generate new completions from the truncation point
    doctor_ready["completion"] = []

    # Store the truncated guess_history in info for environment reconstruction
    # The environment will read this to reconstruct state (mistakes, found_categories, etc.)
    truncated_guess_history = truncated_result.get("guess_history", [])
    if "info" not in doctor_ready:
        doctor_ready["info"] = {}
    doctor_ready["info"]["resumed_from_guess_history"] = truncated_guess_history

    # Clear guess_history - we'll start fresh (it will be reconstructed from info)
    doctor_ready["guess_history"] = []

    return doctor_ready


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare bad examples for doctoring (splits by rejection reason)"
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
    print(f"Output directory: {output_dir}")
    print(f"  Gameplay issues -> {gameplay_output}")
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
    skipped_examples = []

    for example in bad_examples:
        rejection_reason = example.get("rejection_reason")
        if rejection_reason in GAMEPLAY_REJECTION_REASONS:
            gameplay_examples.append(example)
        else:
            skipped_examples.append(example)

    print(f"  Gameplay issues: {len(gameplay_examples)}")
    if skipped_examples:
        print(f"  Skipped (non-gameplay): {len(skipped_examples)}")
    print()

    # Initialize ruleset config
    from connections.rulesets import get_ruleset_config
    ruleset_config = get_ruleset_config("nyt")

    # Initialize openai llm client for ranking categories (only needed for gameplay)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_client = None
    if openai_api_key and gameplay_examples:
        llm_client = OpenAI(api_key=openai_api_key)
    elif gameplay_examples and not openai_api_key:
        print("Warning: OPENAI_API_KEY not set, will use original category order")
    print()

    # Process gameplay examples (truncate + add doctoring instructions)
    if gameplay_examples:
        print("Preparing gameplay doctoring examples...")
        doctor_gameplay_ready = []
        for i, processed_bad_example in enumerate(gameplay_examples):
            if (i + 1) % 10 == 0:
                print(f"  Processing example {i + 1}/{len(gameplay_examples)}...")

            # Get original guess_history from the processed example (it's preserved in the copy)
            original_guess_history = processed_bad_example.get("guess_history", [])
            truncated = truncate_processed_rollout(processed_bad_example, original_guess_history)

            # Create doctoring-ready example with system prompt instructions
            # Pass original_guess_history to preserve the model's natural category ordering
            doctor_ready = create_doctor_ready_example(
                truncated,
                original_guess_history,
                ruleset_config,
                llm_client
            )
            doctor_gameplay_ready.append(doctor_ready)

        # Save gameplay results
        print(f"\nSaving gameplay results...")
        print(f"  {len(doctor_gameplay_ready)} examples -> {gameplay_output}")
        with open(gameplay_output, "w") as f:
            for example in doctor_gameplay_ready:
                f.write(json.dumps(example) + "\n")
    else:
        print("\nNo gameplay examples to process")

    print("\nDone!")


if __name__ == "__main__":
    main()

