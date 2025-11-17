from ..rulesets import RulesetConfig
from ..utils import items_to_string

MISTAKES_COUNT_AT_END_TEMPLATE = "Mistakes will start counting when you have {mistakes_count_when_x_categories_remain} or fewer categories remaining. You can then make {max_mistakes} mistakes."

# Puzzle Info Prompt Parts
TITLE_TEMPLATE = "Title/Hint: {title}"
TAGS_TEMPLATE = "Tags: {tags}"
COUNTRY_TEMPLATE = "Country of Origin: {country}"
CATEGORIES_INFO_TEMPLATE = "There are {total_categories} categories containing {expected_group_size} items each."


def puzzle_info_prompt_part(
    title: str | None,
    tags: list[str] | None,
    country: str | None,
    total_categories: int,
    expected_group_size: int,
) -> str:
    """Generate puzzle info prompt part, conditionally including title and tags."""
    puzzle_info_parts = ["**Puzzle Info**"]

    if title:
        puzzle_info_parts.append(TITLE_TEMPLATE.format(title=title))

    if tags and len(tags) > 0:
        tags_str = ", ".join(tags)
        puzzle_info_parts.append(TAGS_TEMPLATE.format(tags=tags_str))

    if country:
        puzzle_info_parts.append(COUNTRY_TEMPLATE.format(country=country))

    puzzle_info_parts.append(
        CATEGORIES_INFO_TEMPLATE.format(
            total_categories=total_categories,
            expected_group_size=expected_group_size,
        )
    )

    return "\n".join(puzzle_info_parts)


# Current Game State Prompt Parts
CATEGORIES_TEMPLATE = "You have found {found_categories}/{total_categories} categories"
MISTAKES_MADE_TEMPLATE = "You have made {mistakes}/{max_mistakes} mistakes"
REMAINING_ITEMS_TEMPLATE = "Remaining items: {items_str}"


def current_state_prompt_part(
    found_categories: int,
    total_categories: int,
    mistakes: int,
    max_mistakes: int,
    counting_mistakes: bool,
    items: list[str],
) -> str:
    """Generate current state prompt based on found categories, mistakes, and remaining items."""
    current_state_parts = ["**Current Game State**"]
    current_state_parts.append(
        CATEGORIES_TEMPLATE.format(
            found_categories=found_categories, total_categories=total_categories
        )
    )
    if counting_mistakes:
        current_state_parts.append(
            MISTAKES_MADE_TEMPLATE.format(mistakes=mistakes, max_mistakes=max_mistakes)
        )
    current_state_parts.append(
        REMAINING_ITEMS_TEMPLATE.format(items_str=items_to_string(items))
    )
    return "\n".join(current_state_parts)


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

    # Determine if mistakes count right away
    threshold = ruleset_config.mistakes_count_when_x_categories_remain
    mistakes_count_immediately = threshold == "any"

    # Build the base prompt
    prompt_parts = []

    # Add puzzle info
    prompt_parts.append(
        puzzle_info_prompt_part(
            title=title,
            tags=tags,
            country=country,
            total_categories=total_categories,
            expected_group_size=expected_group_size,
        )
    )

    # Add mistake counting note if mistakes don't count immediately
    if not mistakes_count_immediately:
        prompt_parts.append(
            MISTAKES_COUNT_AT_END_TEMPLATE.format(
                mistakes_count_when_x_categories_remain=threshold,
                max_mistakes=ruleset_config.max_mistakes,
            )
        )

    current_state = current_state_prompt_part(
        0,
        total_categories,
        0,
        ruleset_config.max_mistakes,
        mistakes_count_immediately,
        words,
    )

    prompt_parts.append(current_state)

    return "\n".join(prompt_parts).strip()
