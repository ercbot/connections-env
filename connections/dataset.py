import random

from datasets import Dataset

from .prompts import generate_game_start_prompt
from .rulesets import RulesetConfig


def format_puzzle(example: dict, ruleset_config: RulesetConfig) -> dict:
    """
    Format a single raw puzzle into the structure expected by the environment.

    Converts the raw format (group_words, group_themes, etc.) to:
    - question: formatted starting prompt with items
    - info: {categories: list of {group: theme, members: item_list}}

    Args:
        example: Raw puzzle dict from the dataset
        ruleset_config: Configuration for the ruleset

    Returns:
        Dict with 'question' and 'info' keys
    """
    # Convert group_words, group_themes, and group_linking_terms to categories format
    categories = []
    group_linking_terms = example.get("group_linking_terms", [])

    for i, (words_list, theme) in enumerate(
        zip(example["group_words"], example["group_themes"])
    ):
        # Extract linking terms for this category
        linking_terms = []
        if i < len(group_linking_terms):
            # Split linking terms by spaces/commas and clean them
            raw_terms = group_linking_terms[i]
            if isinstance(raw_terms, str):
                # Handle both comma and space separation
                linking_terms = [
                    term.strip()
                    for term in raw_terms.replace(",", " ").split()
                    if term.strip()
                ]

        categories.append(
            {"group": theme, "members": words_list, "linking_terms": linking_terms}
        )

    # Extract puzzle data explicitly
    items = example[
        "all_words"
    ].copy()  # Dataset field name is "all_words" but contains items
    grid_size = example["grid_size"]  # Required field
    num_groups = example["num_groups"]  # Required field
    title = example.get("title")  # Optional
    tags = example.get("tags", [])  # Optional, default to empty list
    puzzle_id = example.get("puzzle_id")
    country = example.get("country")

    # Shuffle items using puzzle_id as seed for deterministic but puzzle-specific ordering
    # This ensures the same puzzle always shuffles the same way, but different puzzles
    # get different shuffle patterns (avoiding the issue where all puzzles might
    # shuffle in the same pattern with a fixed seed)
    seed = int(puzzle_id)
    random.Random(seed).shuffle(items)

    # Generate game start prompt using extracted data
    game_start_prompt = generate_game_start_prompt(
        words=items,  # Parameter name kept for compatibility, but contains items
        grid_size=grid_size,
        num_groups=num_groups,
        ruleset_config=ruleset_config,
        title=title,
        tags=tags,
        country=country,
    )

    return {
        "question": game_start_prompt,
        "info": {
            "categories": categories,
            "puzzle_id": example["puzzle_id"],
            "all_words": items,  # Dataset field name kept for compatibility, stores shuffled item order
        },
    }


def prep_dataset(dataset: Dataset, ruleset_config: RulesetConfig) -> Dataset:
    """
    Prepare a dataset for the Connections environment.

    Applies format_puzzle to each example in the dataset.

    Args:
        dataset: Dataset from HuggingFace
        ruleset_config: Configuration for the ruleset

    Returns:
        Dataset: Formatted dataset
    """
    return dataset.map(
        lambda example: format_puzzle(example, ruleset_config), num_proc=1
    )
