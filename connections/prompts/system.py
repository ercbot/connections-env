from ..rulesets import RulesetConfig


MISTAKE_RULES_ALWAYS_COUNT = """\
- If you don't correctly guess all the words in a category you will use one of your {max_mistakes} mistakes, make {max_mistakes} mistakes and you lose."""


MISTAKE_RULES_COUNT_AT_END = """\
- Early in the game, incorrect guesses don't count as mistakes
- When you have {mistakes_count_when_x_categories_remain} or fewer categories remaining, incorrect guesses start counting as mistakes
- Make {max_mistakes} mistakes during the mistake-counting phase and you lose."""


ONE_AWAY_HINT = """\
- If you correctly guessed most but not all words in a category, you will receive a "One Away" hint but still use one of your mistakes."""


THEMES_REVEALED_IMMEDIATELY = """\
When you correctly identify a category, its theme will be revealed to you."""


THEMES_REVEALED_AT_END = """\
Category themes are not revealed until the end of the game."""


GAME_EXAMPLES = """

Categories Examples (these examples show 4-word groups, but group sizes may vary):
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
<tip>
Group sizes are consistent within each puzzle - all groups in a puzzle will have the same number of words.
</tip>
</tips>

"""


# Word Guessing Instructions
WORD_GUESSING_INSTRUCTIONS = """
How to play:
1. Select words that you think belong together. Your guess can have these outcomes:
- If all words are in the same category, you will complete that group. You now know these words can't be part of any other categories.
{mistake_rules}
{one_away_hint}
2. Repeat until you have found all groups (you win) or make too many mistakes (you lose).{theme_revelation}{game_examples}

You don't have to guess the theme, just the words in the group. The order of the words in the category doesn't matter.

IMPORTANT: Make only ONE guess at a time. Do not include multiple <guess> tags in your response.

Format your guess using XML tags as a list of words. The number of words per group will be specified when the game begins:

<guess>[WORD1, WORD2, WORD3, ...]</guess>"""

CATEGORY_GUESSING_INSTRUCTIONS = """
Once you have successfully guessed all categories (or ran out of mistakes), you'll have a chance to guess the themes for bonus points.

You will see all the categories with their words revealed. Your task is to guess what theme connects the words in each category.

Format your theme guesses using XML tags for each category:

<category_1_guess>your theme guess</category_1_guess>
<category_2_guess>your theme guess</category_2_guess>
... (and so on for each category)

Think about what connects the words - it could be:
- A common category or type
- Words that can follow/precede a common word or phrase
- A shared characteristic or property
- Any other thematic connection

Example:

prompt:
Category 1: [BASS, TROUT, SALMON, TUNA]
Category 2: [ANT, DRILL, ISLAND, OPAL]
Category 3: [STAG, SINGLE, SOLO, FREE]
Category 4: [RAM, BUCK, BILLY, JACK]

answer:
<category_1_guess>Fish</category_1_guess>
<category_2_guess>Starts with fire</category_2_guess>
<category_3_guess>Available</category_3_guess>
<category_4_guess>Male animals</category_4_guess>
"""

MULTI_PHASE_INSTRUCTIONS = """
<round_1>
{word_guessing_instructions}
</round_1>

<round_2>
{category_guessing_instructions}
</round_2>
"""

BASE_INSTRUCTIONS = """\
You are playing a game of *Connections*
The words are grouped into secret categories. The words in each category are connected by a common theme. Your goal is to find the members of the groups.

{game_instructions}
"""


def generate_system_prompt(ruleset_config: RulesetConfig) -> str:
    """Generate system prompt based on ruleset configuration using templates."""

    # Determine mistake rules (generic - specific counts will be in game start prompt)
    threshold = ruleset_config.mistakes_count_when_x_categories_remain
    if threshold == "any":
        mistake_rules = MISTAKE_RULES_ALWAYS_COUNT.format(
            max_mistakes=ruleset_config.max_mistakes
        )
    else:
        mistake_rules = MISTAKE_RULES_COUNT_AT_END.format(
            mistakes_count_when_x_categories_remain=threshold,
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

    # Choose the appropriate instructions based on theme guessing
    if ruleset_config.end_game_theme_guessing:
        # Use multi-phase instructions with round markers
        word_instructions = WORD_GUESSING_INSTRUCTIONS.format(
            mistake_rules=mistake_rules,
            one_away_hint=one_away_hint,
            theme_revelation=theme_revelation,
            game_examples=GAME_EXAMPLES,
        )

        game_instructions = MULTI_PHASE_INSTRUCTIONS.format(
            word_guessing_instructions=word_instructions,
            category_guessing_instructions=CATEGORY_GUESSING_INSTRUCTIONS,
        )
    else:
        # Use single phase instructions
        game_instructions = WORD_GUESSING_INSTRUCTIONS.format(
            mistake_rules=mistake_rules,
            one_away_hint=one_away_hint,
            theme_revelation=theme_revelation,
            game_examples=GAME_EXAMPLES,
        )

    return BASE_INSTRUCTIONS.format(game_instructions=game_instructions)
