"""
Reward functions for the Connections environment.

Module-level @vf.reward-decorated functions discovered by the v1 Taskset.
Receives task and state per the v1 reward contract.
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1 import State, Task

from .utils import GuessRecord


@vf.reward(weight=0.5)
async def valid_guesses(task: Task, state: State) -> float:
    """Proportion of guesses that were valid (not invalid status).

    Valid statuses: correct, auto, incorrect, one_away. Invalid status: invalid.
    """
    _ = task
    history = state.get("guess_history", [])
    if not history:
        return 0.0
    valid = sum(1 for g in history if g["status"] != "invalid")
    return valid / len(history)


@vf.reward(weight=0.5)
async def almost_found_categories(task: Task, state: State) -> float:
    """Count one-away guesses for categories that were never correctly found.

    Avoids double-counting: if a category was guessed one-away then later
    correctly, only the correct guess earns reward.
    """
    _ = task
    history = state.get("guess_history", [])
    correctly_found: set[int] = set()
    one_away: set[int] = set()
    for g_dict in history:
        g = GuessRecord(**g_dict)
        if g.is_correct and g.category_idx is not None:
            correctly_found.add(g.category_idx)
        elif g.is_mistake and g.category_idx is not None:
            one_away.add(g.category_idx)
    return float(len(one_away - correctly_found))


@vf.reward(weight=4.0)
async def found_categories(task: Task, state: State) -> float:
    """Proportion of categories found (0.0–1.0)."""
    total = len(task["info"]["categories"])
    if total == 0:
        return 0.0
    return float(state.get("found_categories", 0)) / total


@vf.reward(weight=1.0)
async def efficiency_bonus(task: Task, state: State) -> float:
    """Reward efficient play: fewer manual guesses to find all categories.

    Perfect play on a 4-category puzzle = 3 manual guesses + 1 auto-completion.
    Efficiency = (total_categories - 1) / actual_manual_guesses, capped at 1.0.
    """
    if state.get("found_categories", 0) == 0:
        return 0.0
    history = state.get("guess_history", [])
    if not history:
        return 0.0
    total = len(task["info"]["categories"])
    min_needed = total - 1
    if min_needed == 0:
        return 1.0
    actual = sum(1 for g in history if g["status"] != "auto")
    if actual == 0:
        return 1.0
    return min(min_needed / actual, 1.0)


REWARDS = [
    valid_guesses,
    almost_found_categories,
    found_categories,
    efficiency_bonus,
]
