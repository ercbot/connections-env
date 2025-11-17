"""
Shared utility functions for Connections game.
"""


def is_theme_match(
    actual_theme: str, guessed_theme: str | None, linking_terms: list[str]
) -> bool:
    """
    Check if a guessed theme matches the actual theme using linking terms for flexibility.

    Args:
        actual_theme: The correct theme from the dataset
        guessed_theme: The user's guess (None if no guess provided)
        linking_terms: List of linking words/phrases that should also count as correct

    Returns:
        True if the guess matches the theme or any linking terms
    """
    if not guessed_theme:
        return False

    # Clean up strings for comparison
    actual_clean = actual_theme.lower().strip()
    guessed_clean = guessed_theme.lower().strip()

    # Exact match
    if actual_clean == guessed_clean:
        return True

    # Check if guess matches any linking terms
    for term in linking_terms:
        term_clean = term.lower().strip()

        # Exact match with linking term
        if guessed_clean == term_clean:
            return True

        # Check if guessed theme contains the linking term (or vice versa)
        if term_clean in guessed_clean or guessed_clean in term_clean:
            return True

    # Check if any word from the guess appears in the actual theme or linking terms
    guessed_words = set(guessed_clean.split())
    actual_words = set(actual_clean.split())

    # Check for word overlap with actual theme
    if guessed_words & actual_words:
        return True

    # Check for word overlap with linking terms
    for term in linking_terms:
        term_words = set(term.lower().split())
        if guessed_words & term_words:
            return True

    return False


def items_to_string(items: list[str]) -> str:
    """
    Format a list of items as a comma-separated string with backticks, wrapped in square brackets.

    Args:
        items: List of items to format (can be words, phrases, abbreviations, fragments, etc.)

    Returns:
        Formatted string like "[`item1`, `item2`, `item3`]"
    """
    formatted = ", ".join(f"`{item}`" for item in items)
    return f"[{formatted}]"

