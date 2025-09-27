from .rulesets import RulesetConfig


BASE_GAME_DESCRIPTION = """\
You are playing a game of *Connections*

The words are grouped into secret categories. The words in each category are connected by a common theme. Your goal is to find the members of the groups."""


MISTAKE_RULES_ALWAYS_COUNT = """\
- If you don't correctly guess all the words in a category you will use one of your {max_mistakes} mistakes, make {max_mistakes} mistakes and you lose."""


MISTAKE_RULES_COUNT_AT_END = """\
- Early in the game, incorrect guesses don't count as mistakes
- When you have {mistakes_start_counting_at} or fewer categories remaining, incorrect guesses start counting as mistakes
- Make {max_mistakes} mistakes during the mistake-counting phase and you lose."""


ONE_AWAY_HINT = """\
- If you correctly guessed most but not all words in a category, you will receive a "One Away" hint but still use one of your mistakes."""


THEMES_REVEALED_IMMEDIATELY = """\
When you correctly identify a category, its theme will be revealed to you."""


THEMES_REVEALED_AT_END = """\
Category themes are not revealed until the end of the game."""


END_GAME_THEME_GUESSING = """

After finding all categories (or running out of mistakes), you'll have a chance to guess the themes for bonus points."""


GAME_EXAMPLES = """

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


BASE_SYSTEM_PROMPT = """{base_game_description}

How to play:
1. Select words that you think belong together. Your guess can have these outcomes:
- If all words are in the same category, you will complete that group. You now know these words can't be part of any other categories.
{mistake_rules}
{one_away_hint}
2. Repeat until you have found all groups (you win) or make too many mistakes (you lose).{theme_revelation}{end_game_rules}{game_examples}

Format your guess using XML tags as a list of words. Make sure to include all words for a complete group:

<guess>[{word_placeholders}]</guess>"""


def generate_system_prompt(
    ruleset_config: RulesetConfig, expected_group_size: int, total_categories: int
) -> str:
    """Generate system prompt based on ruleset configuration using templates."""

    # Determine mistake rules
    if ruleset_config.mistakes_start_counting_at >= total_categories:
        mistake_rules = MISTAKE_RULES_ALWAYS_COUNT.format(
            max_mistakes=ruleset_config.max_mistakes
        )
    else:
        mistake_rules = MISTAKE_RULES_COUNT_AT_END.format(
            mistakes_start_counting_at=ruleset_config.mistakes_start_counting_at,
            max_mistakes=ruleset_config.max_mistakes,
        )

    # One Away hints
    one_away_hint = ONE_AWAY_HINT if ruleset_config.show_one_away_hints else ""

    # Theme revelation
    theme_revelation = (
        THEMES_REVEALED_IMMEDIATELY
        if ruleset_config.reveal_themes_immediately
        else THEMES_REVEALED_AT_END
    )

    # End game rules
    end_game_rules = (
        END_GAME_THEME_GUESSING if ruleset_config.end_game_theme_guessing else ""
    )

    # Word placeholders
    word_placeholders = ", ".join([f"WORD{i + 1}" for i in range(expected_group_size)])

    return BASE_SYSTEM_PROMPT.format(
        base_game_description=BASE_GAME_DESCRIPTION,
        mistake_rules=mistake_rules,
        one_away_hint=one_away_hint,
        theme_revelation=theme_revelation,
        end_game_rules=end_game_rules,
        game_examples=GAME_EXAMPLES,
        word_placeholders=word_placeholders,
    )
