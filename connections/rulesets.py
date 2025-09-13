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
    
    # Base game description
    base_rules = """You are playing a game of *Connections*

The words are grouped into secret categories. The words in each category are connected by a common theme. Your goal is to find the members of the groups."""

    # Dynamic mistake rules
    if ruleset_config.mistakes_start_counting_at >= total_categories:
        # NYT style: mistakes always count
        mistake_text = f"""- If you don't correctly guess all the words in a category you will use one of your {ruleset_config.max_mistakes} mistakes, make {ruleset_config.max_mistakes} mistakes and you lose."""
    else:
        # PuzzGrid style: mistakes only count at end
        mistake_text = f"""- Early in the game, incorrect guesses don't count as mistakes
- When you have {ruleset_config.mistakes_start_counting_at} or fewer categories remaining, incorrect guesses start counting as mistakes
- Make {ruleset_config.max_mistakes} mistakes during the mistake-counting phase and you lose."""

    # One Away hints
    one_away_text = """- If you correctly guessed most but not all words in a category, you will receive a "One Away" hint but still use one of your mistakes.""" if ruleset_config.show_one_away_hints else ""

    # Theme revelation
    theme_text = """

When you correctly identify a category, its theme will be revealed to you.""" if ruleset_config.reveal_themes_immediately else """

Category themes are not revealed until the end of the game."""

    # End game rules
    end_game_text = f"""

After finding all categories (or running out of mistakes), you'll have a chance to guess the themes for bonus points.""" if ruleset_config.end_game_theme_guessing else ""

    # Dynamic word count
    word_placeholders = ", ".join([f"WORD{i+1}" for i in range(expected_group_size)])
    
    # Game examples (keep existing ones)
    examples = """

Categories Examples:
- Theme: FISH, Members: [BASS, TROUT, SALMON, TUNA]
- Theme: FIRE ____, Members: [ANT, DRILL, ISLAND, OPAL]

<tips>
<tip>
Each puzzle has exactly one solution. Watch out for words that seem to belong to multiple categories.

Example:
Words: [RAM, STAG, BILLY, SINGLE, FREE, SOLO, BUCK, JACK]

Answers:
- Theme: Available (Romantically) [STAG, SINGLE, SOLO, FREE]
- Theme: Male Animals [RAM, BUCK, BILLY, JACK]

STAG could be a male animal, but the "Available" group needs it to complete the set, while the male animals can form a complete group without it.
</tip>
<tip>
Categories will always have a more specific theme than "5-letter words", "names", "verbs", etc.
</tip>
</tips>

You don't have to guess the theme, just the words in the group. The order of the words in the category doesn't matter."""

    return f"""{base_rules}

How to play:
1. Select words that you think belong together. Your guess can have these outcomes:
- If all words are in the same category, you will complete that group. You now know these words can't be part of any other categories.
{mistake_text}
{one_away_text}
2. Repeat until you have found all groups (you win) or make too many mistakes (you lose).{theme_text}{end_game_text}{examples}

Format your guess using XML tags as a list of words. Make sure to include all words for a complete group:

<guess>[{word_placeholders}]</guess>"""