# connections

### Overview

- **Environment ID**: `connections`
- **Short description**: Multi-turn word puzzle game where players find groups of related items.
- **Tags**: puzzles, word-games, multi-turn, reasoning

### Dataset

- **Primary dataset**: [`ericbotti/connections-puzzles`](https://huggingface.co/datasets/ericbotti/connections-puzzles)
- **Split sizes**: Train (RL): 7,554 puzzles, Test: 981 puzzles

The dataset includes puzzles scraped from PuzzGrid covering a variety of topics, grid sizes, and difficulty levels. 

### Task

- **Type**: multi-turn, tool-calling
- **Tool**: `guess(items: list[str])` — submit one guess per turn
- **Stop conditions**: `max_mistakes_reached` (4 mistakes), `all_categories_found`, plus base-harness `max_turns_reached` / `prompt_too_long`
- **Ruleset**: NYT only — 4 max mistakes counting from the start, one-away hints enabled, themes revealed immediately on correct guess. 

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval connections
```

Configure model, sampling, and example count:

```bash
uv run vf-eval connections --model gpt-4.1-mini --num-examples 20 --rollouts-per-example 3 --max-tokens 1024 --temperature 0.7
```

### Environment Arguments

Pass via `--extra-env-kwargs '{...}'`:

| Arg                            | Type | Default | Description                                                              |
| ------------------------------ | ---- | ------- | ------------------------------------------------------------------------ |
| `max_turns`                    | int  | `10`    | Maximum model turns per game                                             |
| `is_dataset_raw_puzzles`       | bool | `true`  | If true, run `prep_dataset` on the training set                          |
| `is_eval_dataset_raw_puzzles`  | bool | `true`  | If true, run `prep_dataset` on the eval set                              |
| `system_prompt`                | str  | `None`  | Override the generated system prompt                                     |

`dataset` and `eval_dataset` can be passed programmatically when constructing the env in Python; they default to the train_rl and test splits of `ericbotti/connections-puzzles`.

### Rewards

| Reward                              | Weight | Description                                                |
| ----------------------------------- | ------ | ---------------------------------------------------------- |
| `valid_guesses`                     | 0.5    | Proportion of guesses that passed validation               |
| `almost_found_categories`           | 0.5    | Count of "one away" guesses for categories never found     |
| `found_categories`                  | 4.0    | Proportion of categories found (0.0–1.0)                   |
| `efficiency_bonus`                  | 1.0    | Rewards fewer manual guesses to find all categories        |

### Key behaviors

- **One guess per turn (enforced).** The taskset includes a `@vf.setup` handler that defaults `sampling_args.parallel_tool_calls = False` if neither the harness nor the runner specifies it. vLLM and OpenAI both honor this — vLLM by post-hoc truncation, OpenAI by suppressing parallel emissions at sampling time. Keeps the RL credit-assignment clean (one action → one reward) and prevents games from ending in a single turn when a model emits N parallel mistakes.
- **Auto-completion.** When all but one category has been found, the last is auto-completed and recorded in `guess_history` with `status="auto"`.
- **Resume / doctoring.** v1-native: callers construct a `State` with populated `mistakes` / `found_categories` / `remaining_items` / `guess_history` and pass it to `harness.run(task, state)`, bypassing the `init_game_state` setup.
- **BYO harness.** The `parallel_tool_calls=False` default lives on the taskset (via `@vf.setup`), so any harness composed with `load_taskset()` inherits it: `vf.Env(taskset=load_taskset(), harness=your_harness)`.

### Architecture

```
connections/
├── environment.py    # source generators, guess tool, @vf.setup, @vf.stop, load_environment factory
├── rubric.py         # @vf.reward-decorated reward functions, REWARDS list
├── dataset.py        # prep_dataset — converts raw HF rows to env-shaped tasks
├── prompts/          # system prompt + game-start prompt template
├── rulesets.py       # legacy ruleset config (NYT used; PuzzGrid kept for future revival)
└── utils.py          # GuessRecord dataclass, item formatting helpers
```
