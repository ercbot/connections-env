"""
Shared utility functions for Connections game.
"""

from dataclasses import dataclass
from typing import Literal, Optional


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


def remove_items_one_at_a_time(
    remaining_items: list[str], items_to_remove: list[str]
) -> list[str]:
    """
    Remove items from remaining_items one instance at a time (case-insensitive).

    If an item appears multiple times in remaining_items, only one instance
    is removed for each occurrence in items_to_remove.

    Args:
        remaining_items: List of items to remove from (preserves order)
        items_to_remove: List of items to remove (case-insensitive matching)

    Returns:
        New list with items removed (one instance per item)
    """
    remaining = remaining_items.copy()
    items_to_remove_lower = [item.lower() for item in items_to_remove]

    for item_to_remove_lower in items_to_remove_lower:
        # Find and remove the first matching item (case-insensitive)
        for i, item in enumerate(remaining):
            if item.lower() == item_to_remove_lower:
                remaining.pop(i)
                break

    return remaining


@dataclass
class GuessRecord:
    """Records information about a single guess attempt.

    Status meanings:
    - invalid: Guess failed validation (wrong count, invalid items, or already-found items)
    - incorrect: Valid guess but doesn't match any category
    - one_away: Valid guess that's one item away from matching a category
    - correct: Valid guess that exactly matches a category
    - auto: Category was auto-completed when only one category remained
    """

    items: list[str]  # The items that were guessed
    status: Literal["invalid", "incorrect", "one_away", "correct", "auto"]
    category_idx: Optional[int] = (
        None  # Which category (for one_away, correct, or auto status)
    )
    result_message: Optional[str] = (
        None  # Feedback message for this guess (errors, success messages, etc.)
    )

    @property
    def is_valid(self) -> bool:
        """Returns True if the guess was valid (not invalid)."""
        return self.status != "invalid"

    @property
    def is_correct(self) -> bool:
        """Returns True if the guess was correct or auto-completed."""
        return self.status in ("correct", "auto")

    @property
    def is_mistake(self) -> bool:
        """Returns True if the guess was a mistake (incorrect or one_away)."""
        return self.status in ("incorrect", "one_away")
