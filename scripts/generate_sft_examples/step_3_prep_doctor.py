#!/usr/bin/env python3
"""
Prepare bad examples for doctoring

Args:
- input_file: Path to the bad_examples.jsonl file

For each bad example:
1. Truncate the example to keep only messages up to and including the last correct guess.
   Also stops before any invalid guess appears.
2. Create a doctoring-ready example with a system prompt that instructs which categories to guess:
   - Preserve the order of categories that were already correctly guessed
   - Rank remaining categories by difficulty using GPT-4o
   - Create system prompt with instructions for ALL categories in the optimized order
3. Save to doctor_ready_examples.jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


def get_found_category_indices(guess_history: List[Dict[str, Any]]) -> set:
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


class CategoryRanking(BaseModel):
    """Ranking of categories from easiest to hardest (1-indexed)."""
    ranking: List[int] = Field(description="List of category numbers (1-indexed) in order from easiest to hardest")


def rank_categories_by_difficulty(
    categories: List[Dict[str, Any]],
    llm_client: OpenAI,
    model: str = "gpt-4o"
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

Below are {len(categories)} categories from a Connections puzzle. Your task is to rank them from EASIEST to HARDEST based on how easy it would be for a player to identify the pattern and group the words correctly.

Consider:
- How obvious is the pattern/theme?
- How many potential false connections exist?
- How specialized is the knowledge required?
- How tricky or misleading are the words?

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
    ruleset_config
) -> str:
    """
    Create a system prompt with explicit instructions for which categories to guess in each round.
    
    Args:
        base_system_prompt: The base system prompt from the environment
        all_categories: List of ALL categories to guess (in order)
        ruleset_config: Ruleset configuration
    
    Returns:
        Modified system prompt with doctoring instructions
    """
    if not all_categories:
        return base_system_prompt
    
    # Build category list for the f-string
    category_list = []
    for round_num, category in enumerate(all_categories, start=1):
        theme = category.get("group", "Unknown")
        members = category.get("members", [])
        guess_str = ", ".join(f"`{word}`" for word in members)
        category_list.append(f"{round_num}. <guess>[{guess_str}]</guess>\n  (Theme: {theme})")
    
    categories_text = "\n".join(category_list)
    
    # Build the doctoring instructions as an f-string literal
    doctoring_instructions = f"""
---
## SPECIAL TRAINING DATA GENERATION MODE

You are engaging in a special version of the Connections game environment described above. This special mode is designed to create high-quality training data for supervised fine-tuning (SFT).

### Instructions
- The words to guess for each round are provided to you below.
- However, you must still generate natural, logical reasoning that would lead you to come to the conclusion that the words belong to the category.
- Your reasoning should feel authentic and demonstrate genuine problem-solving thought processes.
- Sometimes the guess given to you will be incorrect, that is okay as it simulates how with these puzzles even correct reasoning can lead to incorrect guesses.
- Do NOT mention any meta-knowledge like "I was told to guess X" or "The prompt indicates the next guess should be Y"

### Example Guess
<think>
I notice that these four words share a common theme...
</think>

<guess>[`WORD1`, `WORD2`, `WORD3`, `WORD4`]</guess>

### Category Guessing Order:
You must guess the categories in the following exact order:

{categories_text}

### Remember:
- Follow the category order exactly as specified above.
- Make exactly one guess per round as specified in the category guessing order.
- Generate natural, logical reasoning for each guess.
- Use the required <think>...</think> and <guess>...</guess> tag format.
- Do NOT reference this meta-instruction or the fact that answers are provided.
- Act as if you are naturally discovering the connections through reasoning.
- Your responses should read as if you are genuinely solving the puzzle.
"""
    
    return base_system_prompt + doctoring_instructions


def truncate_processed_rollout(processed_result: Dict[str, Any], original_guess_history: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def create_doctor_ready_example(
    truncated_result: Dict[str, Any],
    ruleset_config,
    llm_client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Create a doctoring-ready example with a system prompt that instructs which categories to guess.
    
    This replaces the system prompt with one that has explicit instructions for ALL categories:
    1. Categories that were already correctly guessed (in their original order)
    2. Remaining categories (ranked by GPT-4o from easiest to hardest)
    """
    # Get categories and guess history
    categories = truncated_result.get("info", {}).get("categories", [])
    guess_history = truncated_result.get("guess_history", [])
    
    # Get categories that were already correctly guessed, in order
    guessed_categories = get_guessed_categories_in_order(guess_history, categories)
    
    # Determine which categories were already found
    found_indices = get_found_category_indices(guess_history)
    
    # Get remaining categories
    remaining_categories = [
        cat for idx, cat in enumerate(categories)
        if idx not in found_indices
    ]
    
    # Rank remaining categories by difficulty using GPT-4o
    if remaining_categories and llm_client:
        try:
            ranked_remaining = rank_categories_by_difficulty(remaining_categories, llm_client)
        except Exception as e:
            print(f"Warning: Failed to rank categories, using original order: {e}")
            ranked_remaining = remaining_categories
    else:
        # No LLM client or no remaining categories, use original order
        ranked_remaining = remaining_categories
    
    # Combine: guessed categories (in order) + ranked remaining categories
    all_categories_in_order = guessed_categories + ranked_remaining
    
    # If no categories to guess, return as-is (shouldn't happen for doctoring)
    if not all_categories_in_order:
        return truncated_result
    
    # Get the original prompt
    prompt = truncated_result.get("prompt", [])
    
    # Find and replace the system prompt
    new_prompt = []
    base_system_prompt = ""
    
    for msg in prompt:
        if msg.get("role") == "system":
            base_system_prompt = msg.get("content", "")
            # Create new system prompt with doctoring instructions for ALL categories
            new_system_prompt = create_doctoring_system_prompt(
                base_system_prompt,
                all_categories_in_order,
                ruleset_config
            )
            new_prompt.append({
                "role": "system",
                "content": new_system_prompt
            })
        else:
            new_prompt.append(msg)
    
    # Create doctoring-ready example
    doctor_ready = truncated_result.copy()
    doctor_ready["prompt"] = new_prompt
    # Clear the completion - we'll start fresh with the doctoring instructions
    doctor_ready["completion"] = []
    # Clear guess_history - we'll start fresh
    doctor_ready["guess_history"] = []
    
    return doctor_ready


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare bad examples for doctoring"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("bad_examples.jsonl"),
        help="Path to the bad_examples.jsonl file (default: bad_examples.jsonl in current directory)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("doctor_ready_examples.jsonl"),
        help="Output file path (default: doctor_ready_examples.jsonl in current directory)"
    )
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    print(f"Reading from: {input_file}")
    print(f"Output will be written to: {output_file}")
    print()

    # Load bad examples
    bad_examples = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                bad_examples.append(json.loads(line))

    print(f"Loaded {len(bad_examples)} bad examples")

    # Initialize ruleset config
    from connections.rulesets import get_ruleset_config
    ruleset_config = get_ruleset_config("nyt")
    
    # Initialize GPT-4o client for ranking categories
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_client = None
    if openai_api_key:
        llm_client = OpenAI(api_key=openai_api_key)
        print("Using GPT-4o to rank remaining categories by difficulty")
    else:
        print("Warning: OPENAI_API_KEY not set, will use original category order")
    
    print()

    # Process bad examples
    print("Preparing doctoring-ready examples...")
    doctor_ready_examples = []
    for i, processed_bad_example in enumerate(bad_examples):
        if (i + 1) % 10 == 0:
            print(f"  Processing example {i + 1}/{len(bad_examples)}...")
        
        # Get original guess_history from the processed example (it's preserved in the copy)
        original_guess_history = processed_bad_example.get("guess_history", [])
        truncated = truncate_processed_rollout(processed_bad_example, original_guess_history)
        
        # Create doctoring-ready example with system prompt instructions
        doctor_ready = create_doctor_ready_example(truncated, ruleset_config, llm_client)
        doctor_ready_examples.append(doctor_ready)

    # Save results
    print(f"\nSaving results...")
    print(f"  Doctor-ready examples: {len(doctor_ready_examples)} -> {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in doctor_ready_examples:
            f.write(json.dumps(example) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()

