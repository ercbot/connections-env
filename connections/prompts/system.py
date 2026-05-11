from ..rulesets import RulesetConfig

SYSTEM_PROMPT = """\
You are playing *Connections*. Items are grouped by theme. Find the group members.

Select items that belong together:
- All correct = group completed
- Incorrect guess = 1 mistake. {max_mistakes} mistakes = game over.
- All but one member of a group = "One Away" hint, still costs 1 mistake.

Repeat until all groups found (win) or too many mistakes (lose).
Correct guesses reveal the theme immediately.

Examples:
- Theme: FISH, Items: [`BASS`, `TROUT`, `SALMON`, `TUNA`]
- Theme: FIRE ____, Items: [`ANT`, `DRILL`, `ISLAND`, `OPAL`]
- Theme: QUICKLY, Items: [`IN A FLASH`, `AT ONCE`, `RIGHT AWAY`, `IN NO TIME`]

Tips:
- Each puzzle has one solution. Items may fit multiple categories - choose carefully.
- All groups have the same amount of items, specified at game start.
- Items are usually words but can also be phrases, abbreviations, fragments, or emojis. Whatever appears between the backticks is the exact text of the item.

Submit each guess by calling the `guess` tool with the items you believe form a category. Make exactly one `guess` call per turn — you'll see the result before deciding your next guess.
"""


def generate_system_prompt(ruleset_config: RulesetConfig) -> str:
    """Generate the Connections system prompt for the given ruleset.

    NYT-only for now (only `max_mistakes` is parameterised).
    """
    return SYSTEM_PROMPT.format(max_mistakes=ruleset_config.max_mistakes)
