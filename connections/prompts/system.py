from ..rulesets import RulesetConfig

MISTAKE_RULES_ALWAYS_COUNT = """\
- Incorrect guess = 1 mistake. {max_mistakes} mistakes = game over."""


MISTAKE_RULES_COUNT_AT_END = """\
- Early guesses don't count as mistakes
- When {mistakes_count_when_x_categories_remain} or fewer categories remain, mistakes start counting
- {max_mistakes} mistakes during counting phase = game over"""


ONE_AWAY_HINT = """\
- All but one member of a group = "One Away" hint, still costs 1 mistake."""


THEMES_REVEALED_IMMEDIATELY = """\
Correct guesses reveal the theme immediately."""


THEMES_REVEALED_AT_END = """\
Themes revealed only at game end."""


GAME_EXAMPLES = """\
Examples:
- Theme: FISH, Words: [`BASS`, `TROUT`, `SALMON`, `TUNA`]
- Theme: FIRE ____, Words: [`ANT`, `DRILL`, `ISLAND`, `OPAL`]
- Theme: QUICKLY, Words: [`IN A FLASH`, `AT ONCE`, `RIGHT AWAY`, `IN NO TIME`]
"""

GAME_TIPS = """\
Tips:
- Each puzzle has one solution. Words may fit multiple categories - choose carefully.
- All groups have the same amount of words, specified at game start.
"""

# Word Guessing Instructions
WORD_GUESSING_INSTRUCTIONS = """\
Select words that belong together:
- All correct = group completed
{mistake_rules}
{one_away_hint}

Repeat until all groups found (win) or too many mistakes (lose).
{theme_revelation}
{game_examples}
{game_tips}

Note: "word" is a catchall term, sometimes the "words" are whole phrases or other text.
Whatever is between the backticks is the exact text of the "word".

In your response, indicate your guess with this format:
<guess>[`WORD1`, `WORD2`, `WORD3`, ...]</guess>

You can only make one guess per response, but your response can including reasoning or notes.
"""

CATEGORY_GUESSING_INSTRUCTIONS = """\
After finding all word groups, guess each category's theme for bonus points.

Format:
<category_1_guess>theme</category_1_guess>
<category_2_guess>theme</category_2_guess>
(etc.)

Themes can be: common category, words following/preceding a phrase, shared property, or other connection."""

MULTI_PHASE_INSTRUCTIONS = """\
## Round 1
{word_guessing_instructions}

## Round 2
{category_guessing_instructions}
"""

BASE_INSTRUCTIONS = """\
You are playing *Connections*. Words are grouped by theme. Find the group members.

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
            game_tips=GAME_TIPS,
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
            game_tips=GAME_TIPS,
        )

    return BASE_INSTRUCTIONS.format(game_instructions=game_instructions)
