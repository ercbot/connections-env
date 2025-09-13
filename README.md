# connections

### Overview

- **Environment ID**: `connections`
- **Short description**: Multi-turn word puzzle game where players find groups of 4 related words
- **Tags**: puzzles, word-games, multi-turn, reasoning

### Datasets

- **Primary dataset(s)**: `ericbotti/connections-puzzles` - Collection of Connections puzzles with themed word groups from PuzzGrid
- **Source links**: https://huggingface.co/datasets/ericbotti/connections-puzzles
- **Split sizes**: Train: 8,584 puzzles, Test: 954 puzzles

### Task

- **Type**: multi-turn
- **Parser**: XMLParser (custom ConnectionsParser extending XMLParser)
- **Rubric overview**: Rewards valid guesses (0.75x), almost-correct guesses (0.5x), found categories (1.0x each), difficulty bonuses, and efficiency bonuses

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval connections
```

Configure model and sampling:

```bash
uv run vf-eval connections -m gpt-4o-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"max_turns": 10}'
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg            | Type         | Default | Description                                              |
| -------------- | ------------ | ------- | -------------------------------------------------------- |
| `max_turns`    | int          | `10`    | Maximum number of guesses allowed per game               |
| `dataset`      | Dataset      | `None`  | Custom train dataset (if None, loads default HF dataset) |
| `eval_dataset` | Dataset      | `None`  | Custom eval dataset (if None, uses test split from HF)  |

### Metrics

The rubric combines validity and correctness rewards:

| Metric                     | Meaning                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `reward`                   | Weighted sum of all reward functions (max ~8.0 for perfect play)   |
| `guessed_4_words`          | Proportion of guesses with exactly 4 words (0.25x weight)          |
| `proportional_valid_words` | Average proportion of valid words per guess (0.25x weight)         |
| `all_words_valid`          | Proportion of guesses where all 4 words were valid (0.25x weight)  |
| `almost_found_categories`  | Count of 3/4 correct guesses per category (0.5x weight)            |
| `found_categories`         | Total categories successfully found (1.0x weight each)             |
| `difficulty_bonus`         | Bonus based on difficulty levels of found categories (1.0x weight) |
| `efficiency_bonus`         | Reward for finding categories with fewer guesses (1.0x weight)     |
