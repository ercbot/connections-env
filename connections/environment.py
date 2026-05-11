"""
Connections game environment using verifiers.v1 + tool-calling.

NYT ruleset only. PuzzGrid and end-game theme guessing can be re-introduced
later by adding a ruleset abstraction back.

Resume / doctoring: pass a pre-built `State` to `harness.run(task, state)`
instead of relying on task-side replay metadata.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import verifiers.v1 as vf
from datasets import Dataset, load_dataset
from verifiers.v1 import State, Task

from .dataset import prep_dataset
from .prompts import generate_system_prompt
from .prompts.game_start import current_state_prompt_part
from .rubric import REWARDS
from .rulesets import get_ruleset_config
from .utils import GuessRecord, items_to_string, remove_items_one_at_a_time

logger = logging.getLogger(__name__)

# NYT constants. Keep narrow until ruleset abstraction is reintroduced.
MAX_MISTAKES = 4
DEFAULT_MAX_TURNS = 10
DEFAULT_GROUP_SIZE = 4  # fallback only; real value is read per-puzzle

# Default sampling args applied by our Harness. Runner-supplied sampling_args
# (e.g. from rl.toml) overlay on top of these via dict.update, so any key not
# explicitly overridden by the runner persists.
#
# parallel_tool_calls=False: guarantees only one guess per turn. vLLM and
# OpenAI both honor this by truncating the model's response to the first
# tool call (vLLM is post-hoc filter — model still generates the rest, but
# the harness only ever sees one call). This keeps the RL credit-assignment
# clean (one action -> one reward) and prevents games from ending in 2 turns
# when a model emits 4 parallel mistakes.
DEFAULT_SAMPLING_ARGS = {"parallel_tool_calls": False}

_NYT_CONFIG = get_ruleset_config("nyt")


# =============================================================================
# Tool
# =============================================================================


async def guess(items: list[str], state: State, task: Task) -> str:
    """Submit a guess that these items form one of the puzzle's categories.

    Make exactly one `guess` call per turn — you'll see the result before
    deciding your next guess.

    Args:
        items: Exactly 4 items that you believe share a category.
    """
    info = task["info"]
    categories = info["categories"]
    items_lower = {item.lower() for item in items}

    # Group size is per-puzzle (NYT is 4×4, but the dataset also includes
    # 6×4 puzzles). Read from the first category's member count.
    group_size = (
        len(categories[0]["members"]) if categories else DEFAULT_GROUP_SIZE
    )

    # ---- Validation: cheap rejections that don't cost a mistake ----

    if len(items) != group_size:
        msg = (
            f"Invalid guess: you submitted {len(items)} items but need exactly "
            f"{group_size}. This did not cost a mistake; try again."
        )
        _record(state, items, "invalid", None, msg)
        return msg

    all_items_lower = {it.lower() for cat in categories for it in cat["members"]}
    not_in_game = [it for it in items if it.lower() not in all_items_lower]
    if not_in_game:
        label = "item" if len(not_in_game) == 1 else "items"
        verb = "is" if len(not_in_game) == 1 else "are"
        msg = (
            f"Invalid guess: the {label} {items_to_string(not_in_game)} {verb} "
            f"not in the game. This did not cost a mistake; try again."
        )
        _record(state, items, "invalid", None, msg)
        return msg

    remaining_lower = {it.lower() for it in state["remaining_items"]}
    already_found = [it for it in items if it.lower() not in remaining_lower]
    if already_found:
        msg = (
            f"Invalid guess: the items {items_to_string(already_found)} have "
            f"already been guessed. This did not cost a mistake; try again."
        )
        _record(state, items, "invalid", None, msg)
        return msg

    # ---- Match against categories ----

    matched_idx = next(
        (
            idx
            for idx, cat in enumerate(categories)
            if {it.lower() for it in cat["members"]} == items_lower
        ),
        None,
    )

    if matched_idx is not None:
        category = categories[matched_idx]
        state["found_categories"] += 1
        state["remaining_items"] = remove_items_one_at_a_time(
            state["remaining_items"], category["members"]
        )
        msg = (
            f"Correct! Group members: {items_to_string(category['members'])}. "
            f"Theme: {category['group']}."
        )
        _record(state, items, "correct", matched_idx, msg)

        # Auto-complete the last category if exactly one remains.
        if state["found_categories"] == len(categories) - 1:
            for idx, cat in enumerate(categories):
                cat_lower = {it.lower() for it in cat["members"]}
                rem_lower = {it.lower() for it in state["remaining_items"]}
                if cat_lower.issubset(rem_lower):
                    state["found_categories"] += 1
                    state["remaining_items"] = remove_items_one_at_a_time(
                        state["remaining_items"], cat["members"]
                    )
                    auto_msg = (
                        f"All categories found! The last category was: "
                        f"{cat['group']} - {items_to_string(cat['members'])}."
                    )
                    _record(state, cat["members"], "auto", idx, auto_msg)
                    msg = msg + "\n\n" + auto_msg
                    break

        return msg + "\n\n" + _current_state_block(state, task)

    # Incorrect: count a mistake. Detect one-away.
    state["mistakes"] += 1
    one_away_idx = next(
        (
            idx
            for idx, cat in enumerate(categories)
            if len(items_lower & {it.lower() for it in cat["members"]})
            == group_size - 1
        ),
        None,
    )
    if one_away_idx is not None:
        msg = "One away! Incorrect guess."
        _record(state, items, "one_away", one_away_idx, msg)
    else:
        msg = "Incorrect guess."
        _record(state, items, "incorrect", None, msg)
    return msg + "\n\n" + _current_state_block(state, task)


def _record(
    state: State,
    items: list[str],
    status: str,
    category_idx: int | None,
    result_message: str,
) -> None:
    state["guess_history"].append(
        asdict(
            GuessRecord(
                items=list(items),
                status=status,
                category_idx=category_idx,
                result_message=result_message,
            )
        )
    )


def _current_state_block(state: State, task: Task) -> str:
    return current_state_prompt_part(
        found_categories=state["found_categories"],
        total_categories=len(task["info"]["categories"]),
        mistakes=state["mistakes"],
        max_mistakes=MAX_MISTAKES,
        counting_mistakes=True,  # NYT always counts
        items=state["remaining_items"],
    )


# =============================================================================
# Setup + stops
# =============================================================================


@vf.setup
async def init_game_state(task: Task, state: State) -> None:
    """Initialize fresh per-rollout game state.

    Resume is supported via the v1-native path: callers construct a State with
    populated mistakes/found_categories/remaining_items/guess_history and pass
    it directly to `harness.run(task, state)` — bypassing this setup entirely.
    """
    state["mistakes"] = 0
    state["found_categories"] = 0
    state["remaining_items"] = task["info"]["all_words"].copy()
    state["guess_history"] = []


@vf.stop
async def max_mistakes_reached(task: Task, state: State) -> bool:
    _ = task  # required by v1 stop handler signature contract
    return state.get("mistakes", 0) >= MAX_MISTAKES


@vf.stop
async def all_categories_found(task: Task, state: State) -> bool:
    total = len(task["info"]["categories"])
    return total > 0 and state.get("found_categories", 0) >= total


# =============================================================================
# Factories
# =============================================================================


def _row_to_task(row: dict, max_turns: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["question"]}],
        "info": row["info"],
        "max_turns": max_turns,
    }


def load_taskset(
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    system_prompt: str | None = None,
    is_dataset_raw_puzzles: bool = True,
    is_eval_dataset_raw_puzzles: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    if dataset is None:
        dataset = load_dataset("ericbotti/connections-puzzles", split="train_rl")
    if eval_dataset is None:
        eval_dataset = load_dataset("ericbotti/connections-puzzles", split="test")

    if is_dataset_raw_puzzles:
        dataset = prep_dataset(dataset, _NYT_CONFIG)
    if is_eval_dataset_raw_puzzles:
        eval_dataset = prep_dataset(eval_dataset, _NYT_CONFIG)

    sys_prompt = system_prompt or generate_system_prompt(_NYT_CONFIG)

    def source():
        for row in dataset:
            yield _row_to_task(row, max_turns)

    def eval_source():
        for row in eval_dataset:
            yield _row_to_task(row, max_turns)

    return vf.Taskset(
        source=source,
        eval_source=eval_source,
        system_prompt=sys_prompt,
        toolsets=[vf.Toolset(tools=[guess])],
        setups=[init_game_state],
        stops=[max_mistakes_reached, all_categories_found],
        rewards=REWARDS,
        config=config,
    )


def load_environment(
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    system_prompt: str | None = None,
    is_dataset_raw_puzzles: bool = True,
    is_eval_dataset_raw_puzzles: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    **kwargs,
) -> vf.Env:
    """Load the Connections environment (NYT ruleset, v1 + tool-calling).

    Args:
        dataset: Optional pre-loaded training dataset
            (default: HF ericbotti/connections-puzzles train_rl split).
        eval_dataset: Optional pre-loaded eval dataset.
        system_prompt: Optional override for the system prompt.
        is_dataset_raw_puzzles: If True, run prep_dataset on the training set.
        is_eval_dataset_raw_puzzles: If True, run prep_dataset on the eval set.
        max_turns: Maximum model turns per game.
        **kwargs: Forwarded to vf.Env (currently unused).
    """
    taskset = load_taskset(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        is_dataset_raw_puzzles=is_dataset_raw_puzzles,
        is_eval_dataset_raw_puzzles=is_eval_dataset_raw_puzzles,
        max_turns=max_turns,
    )
    harness = vf.Harness(sampling_args=DEFAULT_SAMPLING_ARGS)
    return vf.Env(taskset=taskset, harness=harness)
