from ..rulesets import RulesetConfig
from ..utils import items_to_string

TITLE_TEMPLATE = "Title/Hint: {title}"

TAGS_TEMPLATE = "Tags: {tags}"

COUNTRY_TEMPLATE = "Country: {country}"

GROUP_SIZE_INFO = "Groups: {total_categories} groups of {expected_group_size} each"

MISTAKES_COUNT_AT_END = "Mistakes will start counting when you have {mistakes_count_when_x_categories_remain} or fewer categories remaining. You can then make {max_mistakes} mistakes."

SEPARATOR = "---"

INITIAL_STATE = "You have found 0 Categories"

MISTAKES_INFO = "You have made 0/{max_mistakes}"

REMAINING_ITEMS = "Remaining Items: {items}"


def generate_game_start_prompt(
    words: list[str],  # Keep parameter name for dataset compatibility
    grid_size: str,
    num_groups: int,
    ruleset_config: RulesetConfig,
    title: str | None = None,
    tags: list[str] | None = None,
    country: str | None = None,
) -> str:
    """Generate game start prompt based on ruleset configuration and puzzle specifics."""

    # Parse grid_size to get group info
    total_categories, expected_group_size = map(int, grid_size.split("x"))

    # Use num_groups if it differs from grid_size (for validation/override)
    if num_groups != total_categories:
        total_categories = num_groups

    # Format the items with backticks and brackets
    items_str = items_to_string(words)  # words parameter contains items

    # Determine if mistakes count right away
    threshold = ruleset_config.mistakes_count_when_x_categories_remain
    mistakes_count_immediately = threshold == "any"

    # Build the base prompt
    prompt_parts = []

    # Add title if available
    if title:
        prompt_parts.append(TITLE_TEMPLATE.format(title=title))

    # Add tags if available
    if tags and len(tags) > 0:
        tags_str = ", ".join(tags)
        prompt_parts.append(TAGS_TEMPLATE.format(tags=tags_str))

    # Add country if available
    if country:
        prompt_parts.append(COUNTRY_TEMPLATE.format(country=country))

    # Add group size info
    prompt_parts.append(
        GROUP_SIZE_INFO.format(
            total_categories=total_categories,
            expected_group_size=expected_group_size,
        )
    )

    # Add mistake counting note if mistakes don't count immediately
    if not mistakes_count_immediately:
        prompt_parts.append(
            MISTAKES_COUNT_AT_END.format(
                mistakes_count_when_x_categories_remain=threshold,
                max_mistakes=ruleset_config.max_mistakes,
            )
        )

    # Add separator
    prompt_parts.append(SEPARATOR)

    # Add game state information
    game_state_parts = [
        "",
        INITIAL_STATE,
    ]

    # Only show mistakes info if we're counting mistakes
    if mistakes_count_immediately:
        game_state_parts.append(
            MISTAKES_INFO.format(max_mistakes=ruleset_config.max_mistakes)
        )

    game_state_parts.append(REMAINING_ITEMS.format(items=items_str))
    prompt_parts.extend(game_state_parts)

    return "\n".join(prompt_parts)
