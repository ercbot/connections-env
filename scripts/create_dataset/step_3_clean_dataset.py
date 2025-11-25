"""
Step 3: Clean the scraped dataset before upload

This script applies all cleaning operations to complete_puzzles.jsonl:
1. Remove puzzles with URLs in words
2. Remove image-based puzzles
3. Remove trailing quotes from descriptions and linking_terms
4. Fix backtick typos
5. Fix escaped quotes (\\" -> ")
6. Fix escape sequences (\\t, \\n -> removed)
7. Fix known word typos

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


def fix_escaped_quotes(word):
    """
    Fix escaped quotes in words.

    Many words in the dataset have escaped quotes (\\") that should be
    regular quotes ("). This is a systematic issue from the scraping process.
    """
    if not isinstance(word, str):
        return word

    # Replace escaped quotes with regular quotes
    return word.replace('\\"', '"')


def fix_escape_sequences(word):
    """
    Fix literal escape sequences in words and strip whitespace.

    Some words have literal escape sequences like \\t (tab) and \\n (newline)
    that should be removed. These are typos from the scraping process.
    Also removes leading/trailing whitespace which affects ~12.5% of puzzles.
    """
    if not isinstance(word, str):
        return word

    # Remove literal escape sequences (backslash followed by letter)
    # Common ones: \t (tab), \n (newline), \r (carriage return)
    cleaned = word.replace('\\t', '').replace('\\n', '').replace('\\r', '')
    
    # Strip leading/trailing whitespace (affects ~12.5% of puzzles, mostly trailing)
    return cleaned.strip()


# Manual typo fixes: mapping of puzzle_id -> {incorrect_word: correct_word}
TYPO_FIXES = {
    "35022": {
        "Pain au chocolate": "Pain au chocolat",
    },
    "63016": {
        "Scorched [ ] quake,": "Scorched [ ] quake",
    },
    "27448": {
        "Lail, Moor Bird": "Lail Moor Bird",
    },
    "98081": {
        "Wedded -> Impairedthe appearance": "Wedded -> Impaired the appearance",
    },
    # Add more typos here as they are discovered
    # Format: "puzzle_id": {"incorrect": "correct"}
}


def fix_typos(word, puzzle_id):
    """
    Fix known typos in words based on puzzle ID.

    This is a manual typo fixer that can be extended as more typos
    are discovered in the dataset.

    Args:
        word: The word to check for typos
        puzzle_id: The puzzle ID to look up typo fixes

    Returns:
        The corrected word, or the original word if no fix is needed
    """
    if not isinstance(word, str):
        return word

    # Check if there are typo fixes for this puzzle
    if str(puzzle_id) in TYPO_FIXES:
        typo_map = TYPO_FIXES[str(puzzle_id)]
        # Check if this word matches any known typo
        if word in typo_map:
            return typo_map[word]

    return word


def clean_puzzle(puzzle_data):
    """
    Apply all cleaning operations to a single puzzle.

    Returns:
        tuple: (cleaned_puzzle, modifications_dict)
    """
    puzzle_content = puzzle_data.get("puzzle_content", {})
    puzzle_id = puzzle_data.get("puzzle_id") or puzzle_data.get("id")
    modifications = {"quotes_removed": 0, "backticks_fixed": 0, "escaped_quotes_fixed": 0, "escape_sequences_fixed": 0, "typos_fixed": 0}

    # Clean each group
    cleaned_groups = []
    for group in puzzle_content.get("groups", []):
        # Fix backticks, escaped quotes, and typos in words
        original_words = group.get("words", [])
        cleaned_words = []
        for word in original_words:
            # Fix backticks first
            cleaned_word = fix_backticks(word)
            if cleaned_word != word:
                modifications["backticks_fixed"] += 1
            
            # Then fix escaped quotes
            original_for_escaped_check = cleaned_word
            cleaned_word = fix_escaped_quotes(cleaned_word)
            if cleaned_word != original_for_escaped_check:
                modifications["escaped_quotes_fixed"] += 1
            
            # Fix escape sequences (like \t, \n)
            original_for_escape_check = cleaned_word
            cleaned_word = fix_escape_sequences(cleaned_word)
            if cleaned_word != original_for_escape_check:
                modifications["escape_sequences_fixed"] += 1
            
            # Finally fix typos (after all other fixes, so typo fixes can override if needed)
            original_for_typo_check = cleaned_word
            cleaned_word = fix_typos(cleaned_word, puzzle_id)
            if cleaned_word != original_for_typo_check:
                modifications["typos_fixed"] += 1
            
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
        "escaped_quotes_fixed": 0,
        "escape_sequences_fixed": 0,
        "typos_fixed": 0,
        "valid_output": 0,
    }

    cleaned_puzzles = []

    print("Cleaning puzzles...")
    for puzzle in all_puzzles:
        # Skip failed scrapes
        if puzzle.get("puzzle_content") is None:
            stats["failed_scrape"] += 1
            continue

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
        stats["escaped_quotes_fixed"] += modifications["escaped_quotes_fixed"]
        stats["escape_sequences_fixed"] += modifications["escape_sequences_fixed"]
        stats["typos_fixed"] += modifications["typos_fixed"]
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
    print(f"  Escaped quotes fixed:    {stats['escaped_quotes_fixed']:,}")
    print(f"  Escape sequences fixed:  {stats['escape_sequences_fixed']:,}")
    print(f"  Typos fixed:             {stats['typos_fixed']:,}")
    print()
    print(f"Output: {output_file}")
    print("=" * 80)

    return cleaned_puzzles, stats


if __name__ == "__main__":
    # Run the cleaning
    cleaned_puzzles, stats = clean_dataset()

    print("\nCleaning complete! Ready for step_4_upload_to_hf.py")
