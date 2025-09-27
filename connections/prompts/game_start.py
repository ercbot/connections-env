from ..rulesets import RulesetConfig


TITLE_TEMPLATE = "Puzzle Title: {title}"

WORDS_TEMPLATE = "Words: {words}"

TAGS_TEMPLATE = "Tags: {tags}"

INITIAL_STATE = "You have found 0 categories."

GRID_SIZE_INFO = "Grid size: {grid_size} ({total_categories} groups of {expected_group_size} words each)"

MISTAKES_ALWAYS_COUNT = "You have {max_mistakes} mistakes remaining."

MISTAKES_COUNT_AT_END = "Mistakes will start counting when you have {mistakes_start_counting_at} or fewer categories remaining. You can then make {max_mistakes} mistakes."


def generate_game_start_prompt(
    words: list[str],
    grid_size: str,
    num_groups: int,
    ruleset_config: RulesetConfig,
    title: str | None = None,
    tags: list[str] | None = None
) -> str:
    """Generate game start prompt based on ruleset configuration and puzzle specifics."""

    # Parse grid_size to get group info
    total_categories, expected_group_size = map(int, grid_size.split("x"))

    # Use num_groups if it differs from grid_size (for validation/override)
    if num_groups != total_categories:
        total_categories = num_groups

    # Format the words
    words_str = ", ".join(words)

    # Determine mistake display based on ruleset
    if ruleset_config.mistakes_start_counting_at >= total_categories:
        # Mistakes always count (NYT style)
        mistake_info = MISTAKES_ALWAYS_COUNT.format(
            max_mistakes=ruleset_config.max_mistakes
        )
    else:
        # Mistakes only count later (PuzzGrid style)
        mistake_info = MISTAKES_COUNT_AT_END.format(
            mistakes_start_counting_at=ruleset_config.mistakes_start_counting_at,
            max_mistakes=ruleset_config.max_mistakes
        )

    # Build the base prompt
    prompt_parts = []

    # Add title if available
    if title:
        prompt_parts.append(TITLE_TEMPLATE.format(title=title))

    prompt_parts.append(WORDS_TEMPLATE.format(words=words_str))

    # Add tags if available
    if tags and len(tags) > 0:
        tags_str = ", ".join(tags)
        prompt_parts.append(TAGS_TEMPLATE.format(tags=tags_str))

    prompt_parts.extend([
        "",
        INITIAL_STATE,
        GRID_SIZE_INFO.format(
            grid_size=grid_size,
            total_categories=total_categories,
            expected_group_size=expected_group_size
        ),
        mistake_info
    ])

    return "\n".join(prompt_parts)