from datasets import Dataset

from .prompts import generate_game_start_prompt
from .rulesets import RulesetConfig


def prep_dataset(dataset: Dataset, ruleset_config: RulesetConfig) -> Dataset:
    """
    Internal function to prepare a dataset for the Connections environment.

    Converts the raw dataset format:
    - all_words: list of words
    - group_words: list of word lists
    - group_themes: list of theme descriptions

    To the format expected by the environment:
    - question: formatted starting prompt with words
    - info: {categories: list of {group: theme, members: word_list}}

    Args:
        dataset: Dataset from HuggingFace
        ruleset_config: Configuration for the ruleset

    Returns:
        Dataset: Formatted dataset
    """

    def format_dataset(example):
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
        words = example["all_words"]
        grid_size = example["grid_size"]  # Required field
        num_groups = example["num_groups"]  # Required field
        title = example.get("title")  # Optional
        tags = example.get("tags", [])  # Optional, default to empty list

        # Generate game start prompt using extracted data
        game_start_prompt = generate_game_start_prompt(
            words=words,
            grid_size=grid_size,
            num_groups=num_groups,
            ruleset_config=ruleset_config,
            title=title,
            tags=tags,
        )

        return {
            "question": game_start_prompt,
            "info": {"categories": categories},
        }

    return dataset.map(format_dataset, num_proc=1)
