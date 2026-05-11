from ..rulesets import RulesetConfig
from ..utils import items_to_string

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
    parts = ["**Puzzle Info**"]
    if title:
        parts.append(TITLE_TEMPLATE.format(title=title))
    if tags:
        parts.append(TAGS_TEMPLATE.format(tags=", ".join(tags)))
    if country:
        parts.append(COUNTRY_TEMPLATE.format(country=country))
    parts.append(
        CATEGORIES_INFO_TEMPLATE.format(
            total_categories=total_categories,
            expected_group_size=expected_group_size,
        )
    )
    return "\n".join(parts)


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
    """Generate current state prompt block."""
    parts = [
        "**Current Game State**",
        CATEGORIES_TEMPLATE.format(
            found_categories=found_categories, total_categories=total_categories
        ),
    ]
    if counting_mistakes:
        parts.append(
            MISTAKES_MADE_TEMPLATE.format(mistakes=mistakes, max_mistakes=max_mistakes)
        )
    parts.append(REMAINING_ITEMS_TEMPLATE.format(items_str=items_to_string(items)))
    return "\n".join(parts)


def generate_game_start_prompt(
    words: list[str],  # parameter name kept for dataset compatibility (= items)
    grid_size: str,
    num_groups: int,
    ruleset_config: RulesetConfig,
    title: str | None = None,
    tags: list[str] | None = None,
    country: str | None = None,
) -> str:
    """Generate the per-puzzle game-start prompt.

    NYT ruleset only: mistakes always count from the start, so the
    'mistakes count when X categories remain' branch is omitted.
    """
    total_categories, expected_group_size = map(int, grid_size.split("x"))
    if num_groups != total_categories:
        total_categories = num_groups

    parts = [
        puzzle_info_prompt_part(
            title=title,
            tags=tags,
            country=country,
            total_categories=total_categories,
            expected_group_size=expected_group_size,
        ),
        current_state_prompt_part(
            found_categories=0,
            total_categories=total_categories,
            mistakes=0,
            max_mistakes=ruleset_config.max_mistakes,
            counting_mistakes=True,
            items=words,
        ),
    ]
    return "\n".join(parts).strip()
