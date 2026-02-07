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