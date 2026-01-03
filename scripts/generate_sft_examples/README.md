# SFT Example Generation Pipeline

This pipeline generates supervised fine-tuning (SFT) examples from Connections puzzle game rollouts. It filters, processes, and optionally "doctors" rollouts to create high-quality training examples.

## Quick Start: Iterative Generation (Recommended)

**Script**: `iterative_generate.py`

The simplest way to generate SFT examples is using the iterative generation script, which automatically combines the prep and generation steps in a loop:

```bash
# Run 5 iterations of prep + generate
uv run python iterative_generate.py --loop 5

# Dry-run mode to analyze existing results without running evaluation
uv run python iterative_generate.py --loop 5 --dry-run
```

### How It Works

1. **Load & Analyze**: Loads all existing results from `generate_results/`, deduplicates by puzzle_id (keeping best rollout), and classifies as good/bad
2. **Prep**: Attempts to salvage bad examples by truncating to last valid state
3. **Generate**: Runs evaluation on salvageable examples + any missing puzzles from train_sft
4. **Repeat**: Continues for N loops, accumulating results and building up training data

### Output Format

```
Loop 1: 726 Good, 264 Bad
  - Salvageable: 252
  - Unsalvageable: 12
  - Missing puzzles: 0

Loop 2: 748 Good, 242 Bad | 22 (8.7% of Ran Puzzles) -> Good Examples
  - Salvageable: 231
  - Unsalvageable: 11
  - Missing puzzles: 0
```

**Percentage Calculation**: Shows what % of ran puzzles (salvaged + fresh) were converted to good examples in that loop.

### Key Features

- **Accumulation**: Each loop builds on ALL previous results (not just latest)
- **No Overwrites**: Results saved as `results_N.jsonl` with auto-incrementing N
- **Best Rollout Selection**: Multiple rollouts per puzzle → picks highest quality
- **Smart Resume**: Truncates bad examples to last valid state for continuation
- **Progress Tracking**: Clear statistics showing improvement each loop

### Arguments

- `--loop N`: Number of iterations to run (required)
- `--results-dir PATH`: Directory containing results files (default: `scripts/generate_sft_examples/generate_results`)
- `--dry-run`: Analyze existing results without running evaluation

---

## Pipeline Overview (Manual Steps)

```
┌───────────────────────────────┐
│                               │
│            Step 1:            │
│       Generate Rollouts       │
│                               │
└────────┬──────────────────────┘
         │
         │ results.jsonl
         │
         ▼
┌───────────────────────────────┐
│                               │
│            Step 2:            │
│    Filter Gameplay Issues     │
│   (Invalid/Lost games only)   │
│                               │
└────┬─────────────────────┬────┘
     │                     │
     │ good_examples.      │ bad_examples.
     │ jsonl               │ jsonl
     │                     │
     │                     ▼
     │          ┌──────────────────────┐
     │          │                      │
     │          │       Step 3:        │
     │          │   Prep Gameplay      │
     │          │      Doctoring       │
     │          │                      │
     │          └──────────┬───────────┘
     │                     │
     │                     │ doctor_gameplay.
     │                     │ jsonl
     │                     │
     │                     ▼
     │            ┌─────────────────┐
     │            │                 │
     │            │     Step 4:     │
     │            │ Doctor Gameplay │
     │            │  (Truncation +  │
     │            │   Replay Env)   │
     │            │                 │
     │            └────────┬────────┘
     │                     │
     │                     │ doctored_gameplay.
     │                     │ jsonl
     │                     │
     └────────┬────────────┘
              │
              │ Both streams
              │
              ▼
     ┌────────────────────┐
     │                    │
     │      Step 5:       │
     │   Reduce Tokens    │
     │  (DeepSeek LLM)    │
     │                    │
     └─────────┬──────────┘
               │
               │ good_examples_reduced.jsonl
               │ doctored_gameplay_reduced.jsonl
               │
               ▼
     ┌────────────────────┐
     │                    │
     │      Step 6:       │
     │ Collate & Validate │
     │  Final Dataset     │
     │                    │
     └─────────┬──────────┘
               │
               │ sft_examples.jsonl
               │
               ▼
        Final Dataset
```

## Step 1: Generate Rollouts

**Script**: `step_1_generate.py`

Generates environment rollouts for a set of puzzles by running the Connections game with an LLM agent.

### Usage
```bash
uv run step_1_generate.py <run_name> -n <num_examples> -r <rollouts_per_puzzle> [--puzzle-ids-file FILE]
```

### Arguments
- `run_name`: Directory name for outputs (creates `<run_name>/results.jsonl`)
- `-n`: Number of puzzles to process (-1 for all)
- `-r`: Number of rollouts per puzzle (attempts per puzzle)
- `--puzzle-ids-file`: Optional file containing puzzle IDs to filter (one per line)

### Example
```bash
# Generate 3 rollouts for first 100 puzzles
uv run step_1_generate.py run1 -n 100 -r 3

# Generate rollouts for specific puzzles
uv run step_1_generate.py run2 -n -1 -r 3 --puzzle-ids-file new_puzzles.txt
```

### Output
- `<run_name>/results.jsonl`: Raw rollout results

---

## Step 2: Filter Gameplay Issues

**Script**: `step_2_filter.py`

Groups rollouts by puzzle, selects the best rollout for each puzzle, and classifies them as "good" or "bad" based on gameplay criteria only.

### Quality Criteria

**Good Examples** must pass ALL gameplay checks:
- ✅ No invalid guesses
- ✅ Game won (all categories found)

**Bad Examples** fail one or more gameplay checks and are tagged with a rejection reason:
- `"Invalid Guess"`: Contains invalid guesses
- `"Game Lost"`: Didn't find all categories

**Note**: Token limit checking is now handled in Step 5 (Reduce Tokens)

### How It Works
1. Groups all rollouts by `puzzle_id` (handles combining results from multiple runs)
2. For each puzzle:
   - Sorts rollouts by quality (highest reward, shortest messages)
   - Tests rollouts from best to worst until one passes all checks
   - If none pass, selects best rollout and marks as bad with rejection reason
3. Wraps reasoning in `<think>` tags (preserves guess tags)
4. Outputs statistics tables

### Usage
```bash
uv run step_2_filter.py <input_file>
```

### Example
```bash
uv run step_2_filter.py run1/results.jsonl
```

### Output
- `good_examples.jsonl`: Examples that passed all checks
- `bad_examples.jsonl`: Examples with rejection reasons
- Statistics tables showing token metrics and rejection reason breakdown

---

## Step 3: Prepare for Gameplay Doctoring

**Script**: `step_3_prep_doctor.py`

Prepares bad examples with gameplay issues (invalid guesses or game losses) for doctoring by truncating and ranking categories.

### Processing
- Only processes examples with `"Invalid Guess"` or `"Game Lost"` rejection reasons
- Truncates to last correct guess
- Uses LLM to rank remaining categories by difficulty
- Creates system prompt with optimized category order

### Usage
```bash
uv run step_3_prep_doctor.py bad_examples.jsonl [--output-dir DIR]
```

### Example
```bash
uv run step_3_prep_doctor.py bad_examples.jsonl --output-dir .
```

### Output
- `doctor_gameplay.jsonl`: Examples ready for gameplay doctoring

---

## Step 4: Doctor Gameplay Issues

**Script**: `step_4_doctor_gameplay.py`

Replays the environment with doctoring instructions to generate clean completions for failed examples. Validates results and organizes iterations automatically.

### Strategy
1. Takes truncated examples from Step 3
2. Replays environment with system prompt instructing category order
3. Validates results using same gameplay checks as Step 2:
   - ✅ No invalid guesses
   - ✅ Game won (all categories found)
4. Organizes results in `doctor_results/` directory:
   - Raw environment output → `doctor_results/raw/results.jsonl`
   - Valid results → `doctor_results/doctored_examples_N.jsonl` (N auto-increments)
   - Failed examples → `doctor_gameplay_rerun.jsonl` (for easy retry)

### Usage
```bash
uv run step_4_doctor_gameplay.py [doctor_gameplay.jsonl] [--output-dir DIR] [--rollouts N]
```

### Arguments
- `doctor_gameplay.jsonl`: Input file from Step 3 (or rerun file)
- `--output-dir`: Directory to save results (default: current directory)
- `--rollouts`: Number of rollouts per example (default: 1)

### Example Workflow
```bash
# Initial run - creates doctor_results/doctored_examples_1.jsonl
uv run step_4_doctor_gameplay.py doctor_gameplay.jsonl

# If some failed, rerun them - creates doctor_results/doctored_examples_2.jsonl
uv run step_4_doctor_gameplay.py doctor_gameplay_rerun.jsonl

# Combine all iterations automatically
uv run combine_doctored_results.py

# Result: doctored_gameplay.jsonl (combined from all iterations)
```

### Output Structure
```
.
├── doctor_results/
│   ├── raw/
│   │   └── results.jsonl          # Raw environment output
│   ├── doctored_examples_1.jsonl  # First iteration (valid only)
│   └── doctored_examples_2.jsonl  # Second iteration (valid only)
├── doctor_gameplay_rerun.jsonl     # Failed examples for retry
└── doctored_gameplay.jsonl         # Combined final output (after running combine script)
```

### Validation Statistics
Shows for each run:
- Number of valid examples
- Number of invalid examples
- Breakdown by rejection reason (Invalid Guess, Game Lost)

---

## Step 5: Reduce Tokens

**Script**: `step_5_reduce_tokens.py`

Uses DeepSeek LLM to reduce tokens in ALL examples (both good and doctored gameplay), filtering out examples that still exceed limits after reduction.

### Two-Phase Strategy

**Phase 1: Fix Generation Limit Violations**
- Finds messages exceeding 1024 tokens
- Reduces each to ~80% of limit (819 tokens) for safety margin

**Phase 2: Fix Total Token Violations**
- If total still exceeds limit after Phase 1
- Iteratively reduces longest messages (max 30% per message)
- Targets ~80% of dynamic total limit (varies by puzzle size)

### Safety Mechanisms
- ✅ Extracts only `<think>` content from LLM response (ignores preambles)
- ✅ Verifies exactly one `<think></think>` tag pair
- ✅ Replaces only think content, preserves original `<guess>` structure
- ✅ Validates guess unchanged after reduction
- ✅ Falls back to original if any check fails

### Usage
```bash
# Set DeepSeek API key
export DEEPSEEK_API_KEY="your-key-here"

# Run token reduction
uv run step_5_reduce_tokens.py [good_examples.jsonl] [doctored_gameplay.jsonl] [--output-dir DIR] [--verbose]
```

### Example
```bash
uv run step_5_reduce_tokens.py good_examples.jsonl doctored_gameplay.jsonl --output-dir .
```

### Output
- `good_examples_reduced.jsonl`: Good examples within token limits
- `doctored_gameplay_reduced.jsonl`: Doctored examples within token limits
- Rejection table showing examples that couldn't be reduced enough

### Verbose Mode
Add `--verbose` flag for detailed logging of each reduction step.

---

## Step 6: Collate and Validate Final Dataset

**Script**: `step_6_collate_validate.py`

Combines token-reduced streams into final training dataset with metadata tracking.

### Input Streams
1. `good_examples_reduced.jsonl` → doctoring_type: `"none"`
2. `doctored_gameplay_reduced.jsonl` → doctoring_type: `"gameplay"`

### Processing
1. Loads both input files (already validated and within token limits)
2. Converts to SFT format with metadata:
   - `puzzle_id`: Unique puzzle identifier
   - `doctoring_type`: Which pipeline branch (`"none"` or `"gameplay"`)
   - `reward`: Final reward score
   - `accuracy`: Correct guesses / total guesses
   - `guess_history`: Full history of guesses and outcomes
   - `categories`: Category information (preserved for viewer)
3. Outputs statistics breakdown

### Usage
```bash
uv run step_6_collate_validate.py [good_examples_reduced.jsonl] [doctored_gameplay_reduced.jsonl] [output_file] [--view] [--port 8000]
```

### Example
```bash
uv run step_6_collate_validate.py good_examples_reduced.jsonl doctored_gameplay_reduced.jsonl sft_examples.jsonl --view
```

### Output
- `sft_examples.jsonl`: Final training dataset
- Statistics showing counts and percentages by doctoring type
- Optional web viewer (with `--view` flag)

---

## Utility Scripts

### `combine_doctored_results.py`

Automatically combines all doctoring iterations from the `doctor_results/` directory, deduplicating by puzzle_id and keeping the best result for each puzzle.

```bash
uv run combine_doctored_results.py [--output-dir DIR]
```

**How it works**:
- Finds all `doctored_examples_N.jsonl` files in `doctor_results/`
- Combines them, keeping the highest reward for duplicate puzzle_ids
- Outputs to `doctored_gameplay.jsonl`

**Example**:
```bash
# Combine all iterations in current directory
uv run combine_doctored_results.py

# Specify different directory
uv run combine_doctored_results.py --output-dir /path/to/output
```

**Output**:
- `doctored_gameplay.jsonl`: Combined and deduplicated results from all iterations

### `compare_puzzle_ids.py`

Compares puzzle IDs between datasets to identify new puzzles not in previous runs.

```bash
uv run compare_puzzle_ids.py <input_file> [--output-dir DIR]
```

**Outputs**:
- `new_puzzles.txt`: Puzzle IDs not in train_sft dataset (for use with `--puzzle-ids-file`)
- `previous_run_good_examples.jsonl`: Examples from train_sft matching input puzzles

### `utils.py`

Shared utility functions used across pipeline:
- Token calculation and validation
- `<think>` tag wrapping and preservation
- Rollout processing and truncation
- Category extraction and ordering
- Gameplay validation (`is_valid_guesses_only`, `is_won`)

---

## Complete Example Workflow

```bash
# Step 1: Generate rollouts for new puzzles
uv run compare_puzzle_ids.py run1/total_results.jsonl
uv run step_1_generate.py run2 -n -1 -r 3 --puzzle-ids-file new_puzzles.txt

# Step 2: Filter gameplay issues only
uv run step_2_filter.py run2/results.jsonl

# Step 3: Prepare for gameplay doctoring
uv run step_3_prep_doctor.py bad_examples.jsonl

# Step 4: Doctor gameplay issues (with optional reruns)
export DEEPSEEK_API_KEY="your-key-here"
uv run step_4_doctor_gameplay.py doctor_gameplay.jsonl

# If some failed, rerun them
uv run step_4_doctor_gameplay.py doctor_gameplay_rerun.jsonl

# Combine all doctoring iterations
uv run combine_doctored_results.py

# Step 5: Reduce tokens
uv run step_5_reduce_tokens.py good_examples.jsonl doctored_gameplay.jsonl

# Step 6: Collate and validate final dataset (with optional viewer)
uv run step_6_collate_validate.py good_examples_reduced.jsonl doctored_gameplay_reduced.jsonl sft_examples.jsonl --view
```

---

## Token Limits

| Limit Type | Standard Puzzles (≤16 words) | Large Puzzles (>16 words) |
|------------|------------------------------|---------------------------|
| **Total Tokens** | 2048 | `min(3072, 128 × num_words)` |
| **Generation Tokens** | 1024 | 1024 |

**Doctoring Targets** (80% for safety margin):
- Total: 1638 tokens
- Generation: 819 tokens

---

## Environment Variables

- `DEEPSEEK_API_KEY`: Required for step_4b_doctor_tokens.py

---

## Notes

- **Puzzle Grouping**: Step 2 groups by `puzzle_id` (not `example_id`) to properly handle combined results from multiple runs
- **Think Tags**: All scripts preserve and validate `<think></think>` tag structure
- **Idempotent Processing**: `wrap_reasoning_in_tags()` checks for existing tags before wrapping
- **Quality Preservation**: Doctoring scripts verify guesses remain unchanged
