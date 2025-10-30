#!/usr/bin/env python3
"""
Script to generate prompts to files for different Connections rulesets.
Uses the PrimeIntellect/Qwen3-4B tokenizer to measure token counts.
"""

from pathlib import Path

from transformers import AutoTokenizer

from connections.dataset import prep_dataset
from connections.prompts import generate_system_prompt
from connections.rulesets import RULESETS, get_ruleset_config


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text))


def save_prompt_to_file(ruleset_name: str, config, system_prompt: str, token_count: int):
    """Save system prompt to a markdown file with YAML frontmatter."""
    output_dir = Path("prompts")
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / f"{ruleset_name}.md"

    # Create YAML frontmatter
    frontmatter = f"""---
ruleset: {ruleset_name}
max_mistakes: {config.max_mistakes}
mistakes_count_when_x_categories_remain: {config.mistakes_count_when_x_categories_remain}
show_one_away_hints: {config.show_one_away_hints}
reveal_themes_immediately: {config.reveal_themes_immediately}
end_game_theme_guessing: {config.end_game_theme_guessing}
token_count: {token_count}
---

"""

    content = frontmatter + system_prompt

    with open(filename, "w") as f:
        f.write(content)

    return filename


def main():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-4B")

    # Calculate system prompt tokens for each ruleset
    for ruleset_name in RULESETS.keys():
        ruleset_config = get_ruleset_config(ruleset_name)
        system_prompt = generate_system_prompt(ruleset_config)
        token_count = count_tokens(system_prompt, tokenizer)

        # Save prompt to file
        save_prompt_to_file(ruleset_name, ruleset_config, system_prompt, token_count)

        print(f"{ruleset_name}: {token_count} tokens")


if __name__ == "__main__":
    main()
