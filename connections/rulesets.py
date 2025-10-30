"""
Customizable rulesets functionality which determines scoring and the prompt for the game.

Predfeined rulesets based on the game rules as they are implemented on the NYT website and PuzzGrid.

You can also make custom rulesets from the avaible elements.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Union


@dataclass
class RulesetConfig:
    """Configuration for different Connections game rulesets."""

    max_mistakes: int
    mistakes_count_when_x_categories_remain: Union[
        int, Literal["any"]
    ]  # Number of remaining categories when mistakes start counting, or "any" for always
    show_one_away_hints: bool
    reveal_themes_immediately: bool
    end_game_theme_guessing: bool


RULESETS: Dict[str, RulesetConfig] = {
    "nyt": RulesetConfig(
        max_mistakes=4,
        mistakes_count_when_x_categories_remain="any",  # Always count mistakes (from the beginning)
        show_one_away_hints=True,
        reveal_themes_immediately=True,
        end_game_theme_guessing=False,
    ),
    "puzzgrid": RulesetConfig(
        max_mistakes=3,
        mistakes_count_when_x_categories_remain=2,  # Only start counting when 2 categories left
        show_one_away_hints=False,
        reveal_themes_immediately=False,
        end_game_theme_guessing=True,
    ),
}


def get_ruleset_config(ruleset_name: str) -> RulesetConfig:
    """Get ruleset configuration by name."""
    if ruleset_name not in RULESETS:
        available = ", ".join(RULESETS.keys())
        raise ValueError(f"Unknown ruleset: {ruleset_name}. Available: {available}")
    return RULESETS[ruleset_name]
