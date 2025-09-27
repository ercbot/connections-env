"""
Customizable rulesets functionality which determines scoring and the prompt for the game.

Predfeined rulesets based on the game rules as they are implemented on the NYT website and PuzzGrid.

You can also make custom rulesets from the avaible elements.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RulesetConfig:
    """Configuration for different Connections game rulesets."""
    name: str
    max_mistakes: int
    mistakes_start_counting_at: int  # Number of remaining categories when mistakes start counting
    show_one_away_hints: bool
    reveal_themes_immediately: bool
    end_game_theme_guessing: bool


RULESETS: Dict[str, RulesetConfig] = {
    "nyt": RulesetConfig(
        name="NYT Connections",
        max_mistakes=4,
        mistakes_start_counting_at=4,  # Always count mistakes (from the beginning)
        show_one_away_hints=True,
        reveal_themes_immediately=True,
        end_game_theme_guessing=False
    ),
    "puzzgrid": RulesetConfig(
        name="PuzzGrid/Connections Wall",
        max_mistakes=3,
        mistakes_start_counting_at=2,  # Only start counting when 2 categories left
        show_one_away_hints=False,
        reveal_themes_immediately=False,
        end_game_theme_guessing=True
    )
}


def get_ruleset_config(ruleset_name: str) -> RulesetConfig:
    """Get ruleset configuration by name."""
    if ruleset_name not in RULESETS:
        available = ", ".join(RULESETS.keys())
        raise ValueError(f"Unknown ruleset: {ruleset_name}. Available: {available}")
    return RULESETS[ruleset_name]


def generate_system_prompt(ruleset_config: RulesetConfig, expected_group_size: int, total_categories: int) -> str:
    """Generate system prompt based on ruleset configuration."""
    from .prompts import generate_system_prompt as _generate_system_prompt
    return _generate_system_prompt(ruleset_config, expected_group_size, total_categories)