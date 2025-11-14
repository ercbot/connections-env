---
dataset_info:
  features:
    - name: puzzle_id
      dtype: string
    - name: creator
      dtype: string
    - name: difficulty
      dtype: float64
    - name: quality
      dtype: float64
    - name: country
      dtype: string
    - name: created_at
      dtype: string
    - name: tags
      sequence: string
    - name: title
      dtype: string
    - name: all_words
      sequence: string
    - name: num_groups
      dtype: int64
    - name: group_words
      sequence:
        sequence: string
    - name: group_themes
      sequence: string
    - name: group_linking_terms
      sequence: string
    - name: grid_size
      dtype: string
  splits:
    - name: train_sft
      num_bytes: {sft_num_bytes}
      num_examples: {sft_count}
    - name: train_rl
      num_bytes: {rl_num_bytes}
      num_examples: {rl_count}
    - name: test
      num_bytes: {test_num_bytes}
      num_examples: {test_count}
  download_size: {download_size}
  dataset_size: {dataset_size}
configs:
  - config_name: default
    data_files:
      - split: train_sft
        path: data/train_sft-*
      - split: train_rl
        path: data/train_rl-*
      - split: test
        path: data/test-*
pretty_name: Connections-Like Puzzles
---

# Connections Puzzles Dataset

A high-quality dataset of {total_puzzles:,} puzzle games scraped from PuzzGrid, similar to the popular New York Times Connections game.

Each puzzle has a set of words which the goal is to group into evenly sized categories based on common themes or connections. Additionally the player may guess the category theme for extra points.

## Overview

- **Total Puzzles**: {total_puzzles:,}

- **Data Splits**: Three splits ({sft_pct:.1f}%, {rl_pct:.1f}%, {test_pct:.1f}%), stratified by difficulty rating

  - **train_sft**: {sft_count:,} puzzles, designated for supervised fine-tuning, see [full game examples dataset](https://huggingface.co/datasets/ericbotti/connections-full-games)

  - **train_rl**: {rl_count:,} puzzles for reinforcement learning, see [verifiers rl environment](https://app.primeintellect.ai/dashboard/environments/ericbotti/connections)

  - **test**: {test_count:,} puzzles, for testing/evaluation

- **Quality Filter**: All puzzles have quality rating ≥ 4.0/5.0

- **Average Quality**: {avg_quality:.2f}/5.0

- **Average Difficulty**: {avg_difficulty:.2f}/5.0

- **Creators**: {unique_creators:,} unique puzzle creators

## Dataset Statistics

{stats_table}

## Structure

Each puzzle record contains three categories of information:

### Basic Metadata

- **puzzle_id**: Unique identifier for the puzzle

- **creator**: PuzzGrid username of puzzle creator

- **difficulty**: Difficulty rating (0.0-5.0)

- **quality**: Quality rating (≥4.0 for all puzzles)

- **country**: Country code associated with puzzle (GB, US, AU, etc.)

- **created_at**: Puzzle creation timestamp in UTC (ISO 8601 format)

- **tags**: List of associated tags

### Puzzle Content

- **title**: Puzzle title (if provided)

- **all_words**: Complete list of words to be grouped

- **num_groups**: Number of groups in the puzzle

- **grid_size**: Dimensions (e.g., "4x4", "5x5", "6x6")

### Solution Data

- **group_words**: List of word lists, one for each group

- **group_themes**: Descriptive themes/categories assosciated with each group

- **group_linking_terms**: Keywords/characters used to determine if a player guessed a category correctly.

## Citation

If you use this dataset, please cite:

```
@dataset{{connections_puzzles_2025,
  title={{Connections Puzzles Dataset}},
  author={{Eric Botti}},
  year={{2025}},
  url={{https://huggingface.co/datasets/ericbotti/connections-puzzles}}
}}
```
