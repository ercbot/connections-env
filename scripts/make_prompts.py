#!/usr/bin/env python3
"""
Script to generate prompts to files for different Connections rulesets.
Uses the PrimeIntellect/Qwen3-4B tokenizer to measure token counts.
Generates both system and game_start prompts for each ruleset.
"""

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from connections.prompts import generate_game_start_prompt, generate_system_prompt
from connections.rulesets import RULESETS, get_ruleset_config


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text))


def save_prompt_to_file(
    ruleset_name: str,
    prompt_name: str,
    config,
    prompt_text: str,
    token_count: int,
):
    """Save prompt to a markdown file with YAML frontmatter."""
    output_dir = Path("prompts")
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / f"{ruleset_name}_{prompt_name}.md"

    # Create YAML frontmatter
    frontmatter = f"""---
ruleset: {ruleset_name}
prompt_type: {prompt_name}
max_mistakes: {config.max_mistakes}
mistakes_count_when_x_categories_remain: {config.mistakes_count_when_x_categories_remain}
show_one_away_hints: {config.show_one_away_hints}
reveal_themes_immediately: {config.reveal_themes_immediately}
end_game_theme_guessing: {config.end_game_theme_guessing}
token_count: {token_count}
---

"""

    content = frontmatter + prompt_text

    with open(filename, "w") as f:
        f.write(content)

    return filename


def main():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-4B")

    # Load the first puzzle from train_sft for game_start prompt generation
    print("Loading train_sft dataset...")
    train_sft_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")
    
    if len(train_sft_dataset) == 0:
        print("Error: train_sft dataset is empty")
        return
    
    # Get the first puzzle (raw format)
    first_puzzle = train_sft_dataset[0]
    
    # Extract puzzle data
    words = first_puzzle.get("all_words", [])
    grid_size = first_puzzle.get("grid_size", "")
    num_groups = first_puzzle.get("num_groups", 0)
    title = first_puzzle.get("title")
    tags = first_puzzle.get("tags", [])
    country = first_puzzle.get("country")
    
    print(f"Using first puzzle (ID: {first_puzzle.get('puzzle_id', 'unknown')}) for game_start prompts")
    print()

    # Generate prompts for each ruleset
    for ruleset_name in RULESETS.keys():
        ruleset_config = get_ruleset_config(ruleset_name)
        
        # Generate system prompt
        system_prompt = generate_system_prompt(ruleset_config)
        system_token_count = count_tokens(system_prompt, tokenizer)
        save_prompt_to_file(
            ruleset_name, "system", ruleset_config, system_prompt, system_token_count
        )
        print(f"{ruleset_name}_system: {system_token_count} tokens")
        
        # Generate game_start prompt
        game_start_prompt = generate_game_start_prompt(
            words=words,
            grid_size=grid_size,
            num_groups=num_groups,
            ruleset_config=ruleset_config,
            title=title,
            tags=tags,
            country=country,
        )
        game_start_token_count = count_tokens(game_start_prompt, tokenizer)
        save_prompt_to_file(
            ruleset_name,
            "game_start",
            ruleset_config,
            game_start_prompt,
            game_start_token_count,
        )
        print(f"{ruleset_name}_game_start: {game_start_token_count} tokens")
        print()


if __name__ == "__main__":
    main()
