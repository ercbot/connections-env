# Connections Dataset Creation Pipeline

This document describes the corrected pipeline for creating a clean Connections puzzle dataset.

## Pipeline Steps

### Step 1: Fetch Grid Metadata

**File:** `step_1_fetch_grids.py`

Fetches puzzle metadata from puzzgrid.com API:

- Puzzle IDs
- Creator info
- Difficulty/quality ratings
- Tags and timestamps

We filter these by quality >= 4.0

**Output:** `grids_metadata.jsonl`

### Step 2: Scrape Puzzle Content

**File:** `step_2_scrape_puzzles.py`

Scrapes full puzzle content for each grid ID:

- Decodes UTF-16 based Caesar cipher-encoded puzzle data
- Extracts words, descriptions, and linking terms

**Output:** `complete_puzzles.jsonl` (raw scraped data)

### Step 3: Clean Dataset

**File:** `step_3_clean_dataset.py`

Applies all cleaning operations:

1. **Remove failed scrapes** - puzzles with scrape errors (should be zero)
2. **Remove image puzzles** - puzzles using image URLs
3. **Remove URL words** - puzzles with other URLs (there are a few audio puzzles)
4. **Clean trailing quotes** - from descriptions and linking terms
5. **Fix backticks** - remove ` as we use this as environment, only backticks scraped were typos anyway

**Output:** `complete_puzzles_cleaned.jsonl`

### Step 4: Upload to Hugging Face

**File:** `step_4_upload_to_hf.py`

Prepares and uploads dataset:

- Converts JSONL to HuggingFace Dataset format
- **Creates stratified 3-way split:**
  - `train_sft`: 10% (for supervised fine-tuning)
  - `train_rl`: 80% (for reinforcement learning)
  - `test`: 10% (for evaluation)
- **Stratification by difficulty bins** (0-2, 2-3, 3-4, 4-4.5, 4.5-5)
  - Ensures balanced difficulty across all splits
  - Important for fair evaluation and training
- Split assignment uses a random hash to ensure consisent splits if reuploading dataset
- Uploads to `ericbotti/connections-puzzles`
- Updates readme with dataset stats

## Running the Pipeline

```bash
cd scripts/dataset_creation

# Step 1: Fetch metadata
uv run python3 step_1_fetch_grids.py

# Step 2: Scrape puzzles
uv run python3 step_2_scrape_puzzles.py

# Step 3: Clean dataset
uv run python3 step_3_clean_dataset.py

# Step 4: Upload to Hugging Face
uv run python3 step_4_upload_to_hf.py
```