"""
Step 3: Clean the scraped dataset before upload

This script applies all cleaning operations to complete_puzzles.jsonl:
1. Remove puzzles with URLs in words
2. Remove image-based puzzles
3. Remove trailing quotes from descriptions and linking_terms
4. Fix backtick typos
5. Filter by quality (≥ 4.0)

Input: complete_puzzles.jsonl (raw scraped data from step_2)
Output: complete_puzzles_cleaned.jsonl (ready for upload in step_4)
"""

import json
from pathlib import Path
from collections import Counter


def is_image_puzzle(puzzle_data):
    """Check if puzzle is image-based"""
    # Check tags for "image" tag
    if "image" in [tag.lower() for tag in puzzle_data.get("tags", [])]:
        return True

    # Check if any words are image URLs
    puzzle_content = puzzle_data.get("puzzle_content", {})
    for group in puzzle_content.get("groups", []):
        for word in group.get("words", []):
            if isinstance(word, str) and (
                word.startswith("https://puzzgrid.nyc3.digitaloceanspaces.com/")
                or word.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
                or "digitaloceanspaces.com" in word
            ):
                return True

    return False


def has_url_words(puzzle_data):
    """Check's for URL based puzzles. e.g. audio puzzles"""
    puzzle_content = puzzle_data.get("puzzle_content", {})
    for group in puzzle_content.get("groups", []):
        for word in group.get("words", []):
            if isinstance(word, str) and ("http" in word.lower() or "://" in word):
                return True
    return False


def clean_trailing_quotes(text):
    """Remove trailing quotes and whitespace"""
    if not isinstance(text, str):
        return text

    cleaned = text.rstrip()
    if cleaned.endswith('"'):
        cleaned = cleaned.rstrip('"')

    return cleaned


def fix_backticks(word):
    """
    Remove backticks from words.

    Backticks are used as word wrappers in the Connections environment,
    so they cannot appear within the words themselves. The few instances
    found in the dataset (only 2 total) are typos and should be removed.
    """
    if not isinstance(word, str):
        return word

    # Remove backticks (conflicts with environment's word wrapper syntax)
    return word.replace("`", "")


def clean_puzzle(puzzle_data):
    """
    Apply all cleaning operations to a single puzzle.

    Returns:
        tuple: (cleaned_puzzle, modifications_dict)
    """
    puzzle_content = puzzle_data.get("puzzle_content", {})
    modifications = {"quotes_removed": 0, "backticks_fixed": 0}

    # Clean each group
    cleaned_groups = []
    for group in puzzle_content.get("groups", []):
        # Fix backticks in words
        original_words = group.get("words", [])
        cleaned_words = []
        for word in original_words:
            cleaned_word = fix_backticks(word)
            if cleaned_word != word:
                modifications["backticks_fixed"] += 1
            cleaned_words.append(cleaned_word)

        # Clean description (group theme)
        original_desc = group.get("description", "")
        cleaned_desc = clean_trailing_quotes(original_desc)
        if cleaned_desc != original_desc:
            modifications["quotes_removed"] += 1

        # Clean linking terms
        original_terms = group.get("linking_terms", "")
        cleaned_terms = clean_trailing_quotes(original_terms)
        if cleaned_terms != original_terms:
            modifications["quotes_removed"] += 1

        cleaned_groups.append(
            {
                "words": cleaned_words,
                "description": cleaned_desc,
                "linking_terms": cleaned_terms,
            }
        )

    # Update puzzle with cleaned data
    cleaned_puzzle = puzzle_data.copy()
    cleaned_puzzle["puzzle_content"] = {
        "title": puzzle_content.get("title", ""),
        "groups": cleaned_groups,
    }

    return cleaned_puzzle, modifications


def clean_dataset(
    input_file="complete_puzzles.jsonl", output_file="complete_puzzles_cleaned.jsonl"
):
    """
    Clean the scraped dataset.

    Note: Quality filtering (>= 4.0) already happened in step_1_fetch_grids.py,
    so all puzzles here should already meet the quality threshold.

    Args:
        input_file: Input JSONL file with raw scraped puzzles
        output_file: Output JSONL file with cleaned puzzles
    """

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: {input_file} not found!")
        return

    print("=" * 80)
    print("CLEANING DATASET")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Load all puzzles
    all_puzzles = []
    print("Loading puzzles...")
    with open(input_path, "r") as f:
        for line in f:
            puzzle = json.loads(line.strip())
            all_puzzles.append(puzzle)

    print(f"Loaded {len(all_puzzles)} puzzles")
    print()

    # Track statistics
    stats = {
        "total_input": len(all_puzzles),
        "failed_scrape": 0,
        "image_based": 0,
        "has_urls": 0,
        "quotes_removed": 0,
        "backticks_fixed": 0,
        "valid_output": 0,
    }

    cleaned_puzzles = []

    print("Cleaning puzzles...")
    for puzzle in all_puzzles:
        # Skip failed scrapes
        if puzzle.get("puzzle_content") is None:
            stats["failed_scrape"] += 1
            continue

        # Note: Quality filtering (>= 4.0) already done in step_1

        # Skip image-based puzzles
        if is_image_puzzle(puzzle):
            stats["image_based"] += 1
            continue

        # Skip puzzles with URLs in words
        if has_url_words(puzzle):
            stats["has_urls"] += 1
            continue

        # Apply cleaning operations
        cleaned_puzzle, modifications = clean_puzzle(puzzle)

        stats["quotes_removed"] += modifications["quotes_removed"]
        stats["backticks_fixed"] += modifications["backticks_fixed"]
        stats["valid_output"] += 1

        cleaned_puzzles.append(cleaned_puzzle)

    # Write cleaned dataset
    output_path = Path(output_file)
    print(f"\nWriting cleaned dataset to {output_file}...")
    with open(output_path, "w") as f:
        for puzzle in cleaned_puzzles:
            f.write(json.dumps(puzzle) + "\n")

    # Print statistics
    print()
    print("=" * 80)
    print("CLEANING STATISTICS")
    print("=" * 80)
    print(f"Input puzzles:           {stats['total_input']:,}")
    print(f"  Failed scrapes:        {stats['failed_scrape']:,}")
    print(f"  Image-based:           {stats['image_based']:,}")
    print(f"  Contains URLs:         {stats['has_urls']:,}")
    print()
    print(f"Valid puzzles:           {stats['valid_output']:,}")
    print()
    print(f"Cleaning operations:")
    print(f"  Trailing quotes removed: {stats['quotes_removed']:,}")
    print(f"  Backticks fixed:         {stats['backticks_fixed']:,}")
    print()
    print(f"Output: {output_file}")
    print("=" * 80)

    return cleaned_puzzles, stats


if __name__ == "__main__":
    # Run the cleaning
    cleaned_puzzles, stats = clean_dataset()

    print("\nCleaning complete! Ready for step_4_upload_to_hf.py")
