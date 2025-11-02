# connections

### Overview

- **Environment ID**: `connections`
- **Short description**: Multi-turn word puzzle game where players find groups of 4 related words
- **Tags**: puzzles, word-games, multi-turn, reasoning

### Datasets

- **Primary dataset(s)**: `ericbotti/connections-puzzles` - Collection of Connections puzzles from PuzzGrid
- **Source links**: https://huggingface.co/datasets/ericbotti/connections-puzzles
- **Split sizes**: Train: 8,572 puzzles, Test: 953 puzzles

### Task

- **Type**: multi-turn
- **Parser**: XMLParser (custom ConnectionsParser extending XMLParser)
- **Rubric overview**: Rewards valid guesses (0.5x), almost-correct guesses (0.5x), found categories (4.0x), and efficiency bonus (1.0x)

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval connections
```

Configure model and sampling:

```bash
uv run vf-eval connections -m gpt-4o-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"ruleset": "nyt"}'
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg            | Type            | Default | Description                                              |
| -------------- | --------------- | ------- | -------------------------------------------------------- |
| `ruleset`      | str             | `"nyt"` | Game ruleset: `"nyt"` or `"puzzgrid"` (see Rulesets)     |
| `max_turns`    | int             | `20`    | Maximum number of conversation turns allowed per game    |
| `dataset`      | Dataset \| None | `None`  | Custom train dataset (if None, loads default HF dataset) |
| `eval_dataset` | Dataset \| None | `None`  | Custom eval dataset (if None, uses test split from HF)   |

### Rulesets

Two predefined rulesets are available, based on different puzzle platforms:

**NYT (New York Times) - Default**

```bash
-a '{"ruleset": "nyt"}'
```

- 4 max mistakes
- Mistakes always count (from the start)
- Shows "one away" hints (3 out of 4 correct)
- Reveals category themes immediately
- No end-game theme guessing phase

**PuzzGrid**

```bash
-a '{"ruleset": "puzzgrid"}'
```

- 3 max mistakes
- Mistakes only count when 2 categories remain
- No "one away" hints
- Themes revealed at end
- Has end-game theme guessing bonus round

### Metrics

The rubric rewards both accuracy and efficiency:

| Metric                    | Weight | Description                                         |
| ------------------------- | ------ | --------------------------------------------------- |
| `reward`                  | -      | Total score (max 5.5 for perfect word-phase play)   |
| `valid_guesses`           | 0.5    | Proportion of guesses that were valid (not invalid) |
| `almost_found_categories` | 0.5    | Count of "one away" guesses for unfound categories  |
| `found_categories`        | 4.0    | Proportion of categories found (0.0-1.0)            |
| `efficiency_bonus`        | 1.0    | Reward for finding categories with fewer guesses    |

**Theme Guessing Metrics** (only for `puzzgrid` ruleset):

| Metric                             | Weight | Description                            |
| ---------------------------------- | ------ | -------------------------------------- |
| `attempted_theme_guessing`         | 0.25   | Attempted to guess themes              |
| `guessed_correct_number_of_themes` | 0.5    | Provided correct number of guesses     |
| `found_themes`                     | 4.0    | Proportion of themes correctly guessed |
| `found_all_themes_bonus`           | 1.0    | Bonus for guessing all themes          |

### Key Features

- When all but one category has been found, the last is auto-completed.
- All the AI's guesses (including invalid ones) are tracked in the "guess_history" state variable

### Example Usage

**Evaluate on test set with NYT ruleset:**

```bash
uv run vf-eval connections --model gpt-4.1-mini --rollouts 3
```

**Use PuzzGrid ruleset with theme guessing:**

```bash
uv run vf-eval connections --model gpt-4o --env-args '{"ruleset": "puzzgrid"}' --rollouts 5
```

**Custom max turns:**

```bash
uv run vf-eval connections --model claude-3-5-sonnet --env-args '{"max_turns": 15}' --rollouts 3
```
