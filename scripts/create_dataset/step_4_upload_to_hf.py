import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import xxhash
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

HASH_SEED = 97821  # my favorite number btw


def prepare_connections_dataset(input_file="complete_puzzles_cleaned.jsonl"):
    """
    Convert JSONL to structured format for Hugging Face.

    Note: Input should be the cleaned dataset from step_3_clean_dataset.py
    """

    puzzles = []

    print("Loading puzzles...")
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # All filtering should have been done in step_3
            # Just verify puzzle_content exists (defensive check)
            if data.get("puzzle_content") is None:
                print(
                    f"Warning: Puzzle {data.get('id', 'unknown')} has no puzzle_content, skipping"
                )
                continue

            puzzle_content = data["puzzle_content"]

            # Extract all words and groups
            all_words = []
            groups = []

            for i, group in enumerate(puzzle_content.get("groups", [])):
                words = [w.strip() for w in group.get("words", [])]
                all_words.extend(words)

                groups.append(
                    {
                        "words": words,
                        "theme": group.get("description", "").strip(),
                        "linking_terms": group.get("linking_terms", "").strip(),
                        "group_id": i,
                    }
                )

            title = puzzle_content.get("title", "")

            # Convert Unix timestamp to UTC ISO format
            raw_date = data.get("date", "")
            readable_date = ""
            if raw_date:
                try:
                    # Convert Unix timestamp to UTC ISO format with Z suffix
                    timestamp = int(raw_date)
                    readable_date = datetime.fromtimestamp(
                        timestamp, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid date format: {raw_date}")

            # Create structured puzzle entry
            puzzle_entry = {
                # Basic metadata
                "puzzle_id": str(data["id"]),  # Convert to string
                "creator": data.get("creator", ""),
                "difficulty": float(data.get("difficulty", 0)),
                "quality": float(data.get("quality", 0)),
                "country": data.get("country", ""),
                "created_at": readable_date,
                "tags": data.get("tags", []),
                # Puzzle content
                "title": title if title else "",
                "all_words": all_words,
                "num_groups": len(groups),
                # Group information
                "group_words": [g["words"] for g in groups],
                "group_themes": [g["theme"] for g in groups],
                "group_linking_terms": [g["linking_terms"] for g in groups],
                # Additional metadata
                "grid_size": f"{len(groups)}x{len(all_words) // len(groups) if groups else 0}",
            }

            puzzles.append(puzzle_entry)

    print(f"Processed {len(puzzles)} puzzles from cleaned dataset")
    return puzzles


def get_difficulty_bin(difficulty):
    """
    Assign puzzle to difficulty bin for stratified sampling.
    Bins: 0-2, 2-3, 3-4, 4-4.5, 4.5-5.0
    """
    if difficulty < 2.0:
        return "0-2"
    elif difficulty < 3.0:
        return "2-3"
    elif difficulty < 4.0:
        return "3-4"
    elif difficulty < 4.5:
        return "4-4.5"
    else:
        return "4.5-5.0"


def get_split_assignment(puzzle_id, difficulty, sft_ratio=0.1, rl_ratio=0.8):
    """
    Deterministic hash-based split assignment with stratification.

    Uses xxhash for fast, deterministic assignment based on:
    - puzzle_id: Unique identifier
    - difficulty_bin: Ensures stratification across difficulty levels
    - hash_seed: Fixed seed for reproducibility (searched to overlap with previous random split)

    Args:
        puzzle_id: String puzzle ID
        difficulty: Float difficulty rating
        sft_ratio: Ratio for train_sft (default 0.1 = 10%)
        rl_ratio: Ratio for train_rl (default 0.8 = 80%)

    Returns:
        str: "train_sft", "train_rl", or "test"
    """
    difficulty_bin = get_difficulty_bin(difficulty)
    combined = f"{puzzle_id}_{difficulty_bin}_{HASH_SEED}"
    hash_val = xxhash.xxh64(combined).intdigest()
    normalized = (hash_val % 10000) / 10000.0

    if normalized < sft_ratio:
        return "train_sft"
    elif normalized < sft_ratio + rl_ratio:
        return "train_rl"
    else:
        return "test"


def calculate_split_stats(splits):
    """Calculate statistics for each split."""
    stats = {}

    for split_name, puzzles in splits.items():
        if not puzzles:
            stats[split_name] = {
                "count": 0,
                "avg_difficulty": 0.0,
                "avg_quality": 0.0,
                "difficulty_bins": Counter(),
                "countries": Counter(),
                "creators": set(),
                "grid_sizes": Counter(),
            }
            continue

        difficulties = [p["difficulty"] for p in puzzles]
        qualities = [p["quality"] for p in puzzles]
        countries = Counter(p["country"] for p in puzzles if p.get("country"))
        creators = set(p["creator"] for p in puzzles if p.get("creator"))
        grid_sizes = Counter(p["grid_size"] for p in puzzles if p.get("grid_size"))
        difficulty_bins = Counter(get_difficulty_bin(p["difficulty"]) for p in puzzles)

        stats[split_name] = {
            "count": len(puzzles),
            "avg_difficulty": sum(difficulties) / len(difficulties)
            if difficulties
            else 0.0,
            "avg_quality": sum(qualities) / len(qualities) if qualities else 0.0,
            "difficulty_bins": difficulty_bins,
            "countries": countries,
            "creators": creators,
            "grid_sizes": grid_sizes,
        }

    return stats


def print_stats_table(splits, stats, total_puzzles):
    """Print a formatted table with breakdown stats for each split."""

    print("\n" + "=" * 100)
    print("DATASET SPLIT STATISTICS")
    print("=" * 100)

    # Header row
    header = (
        f"{'Metric':<30} {'Train SFT':<20} {'Train RL':<20} {'Test':<20} {'Total':<15}"
    )
    print(header)
    print("-" * 100)

    # Count row
    sft_count = stats["train_sft"]["count"]
    rl_count = stats["train_rl"]["count"]
    test_count = stats["test"]["count"]
    print(
        f"{'Count':<30} {sft_count:<20,} {rl_count:<20,} {test_count:<20,} {total_puzzles:<15,}"
    )

    # Percentage row
    sft_pct = (sft_count / total_puzzles * 100) if total_puzzles > 0 else 0
    rl_pct = (rl_count / total_puzzles * 100) if total_puzzles > 0 else 0
    test_pct = (test_count / total_puzzles * 100) if total_puzzles > 0 else 0
    total_pct = 100.0
    print(
        f"{'Percentage':<30} {f'{sft_pct:.1f}%':<20} {f'{rl_pct:.1f}%':<20} {f'{test_pct:.1f}%':<20} {f'{total_pct:.1f}%':<15}"
    )

    # Average difficulty
    avg_diff_sft = stats["train_sft"]["avg_difficulty"]
    avg_diff_rl = stats["train_rl"]["avg_difficulty"]
    avg_diff_test = stats["test"]["avg_difficulty"]
    avg_diff_total = (
        sum(
            p["difficulty"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    print(
        f"{'Avg Difficulty':<30} {avg_diff_sft:<20.2f} {avg_diff_rl:<20.2f} {avg_diff_test:<20.2f} {avg_diff_total:<15.2f}"
    )

    # Average quality
    avg_qual_sft = stats["train_sft"]["avg_quality"]
    avg_qual_rl = stats["train_rl"]["avg_quality"]
    avg_qual_test = stats["test"]["avg_quality"]
    avg_qual_total = (
        sum(
            p["quality"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    print(
        f"{'Avg Quality':<30} {avg_qual_sft:<20.2f} {avg_qual_rl:<20.2f} {avg_qual_test:<20.2f} {avg_qual_total:<15.2f}"
    )

    # Unique creators
    unique_creators_sft = len(stats["train_sft"]["creators"])
    unique_creators_rl = len(stats["train_rl"]["creators"])
    unique_creators_test = len(stats["test"]["creators"])
    unique_creators_total = len(
        set().union(*[stats[s]["creators"] for s in ["train_sft", "train_rl", "test"]])
    )
    print(
        f"{'Unique Creators':<30} {unique_creators_sft:<20,} {unique_creators_rl:<20,} {unique_creators_test:<20,} {unique_creators_total:<15,}"
    )

    print("-" * 100)

    # Difficulty bin breakdown
    all_bins = set()
    for split_stats in stats.values():
        all_bins.update(split_stats["difficulty_bins"].keys())

    for bin_key in sorted(all_bins):
        sft_bin = stats["train_sft"]["difficulty_bins"].get(bin_key, 0)
        rl_bin = stats["train_rl"]["difficulty_bins"].get(bin_key, 0)
        test_bin = stats["test"]["difficulty_bins"].get(bin_key, 0)
        total_bin = sft_bin + rl_bin + test_bin
        print(
            f"{'Difficulty ' + bin_key:<30} {sft_bin:<20,} {rl_bin:<20,} {test_bin:<20,} {total_bin:<15,}"
        )

    print("-" * 100)

    # Top countries
    all_countries = Counter()
    for split_stats in stats.values():
        all_countries.update(split_stats["countries"])

    for country, total_count in all_countries.most_common(5):
        sft_country = stats["train_sft"]["countries"].get(country, 0)
        rl_country = stats["train_rl"]["countries"].get(country, 0)
        test_country = stats["test"]["countries"].get(country, 0)
        print(
            f"{'Country ' + country:<30} {sft_country:<20,} {rl_country:<20,} {test_country:<20,} {total_count:<15,}"
        )

    print("=" * 100)


def create_stratified_splits(puzzles, sft_ratio=0.1, rl_ratio=0.8):
    """
    Create deterministic stratified 3-way split using hash-based assignment.

    The hash-based approach ensures:
    - Deterministic splits (same input always produces same output)
    - Stratification by difficulty (each difficulty bin split proportionally)
    - No need for random seeds or shuffling
    - Reproducibility across different systems

    Args:
        puzzles: List of puzzle dictionaries
        sft_ratio: Ratio for train_sft (default 0.1 = 10%)
        rl_ratio: Ratio for train_rl (default 0.8 = 80%)

    Returns:
        dict: {"train_sft": [...], "train_rl": [...], "test": [...]}
    """
    # Assign each puzzle to a split
    train_sft = []
    train_rl = []
    test = []

    for puzzle in puzzles:
        puzzle_id = str(puzzle["puzzle_id"])
        difficulty = puzzle.get("difficulty", 0.0)

        # Get deterministic split assignment
        split = get_split_assignment(puzzle_id, difficulty, sft_ratio, rl_ratio)

        # Add to appropriate split
        if split == "train_sft":
            train_sft.append(puzzle)
        elif split == "train_rl":
            train_rl.append(puzzle)
        else:
            test.append(puzzle)

    return {"train_sft": train_sft, "train_rl": train_rl, "test": test}


def write_splits_to_jsonl(splits, output_dir="."):
    """Write each split to a separate JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, puzzles in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for puzzle in puzzles:
                f.write(json.dumps(puzzle) + "\n")


def generate_stats_table_markdown(splits, stats, total_puzzles):
    """Generate markdown table with statistics for README."""
    lines = []
    lines.append("| Metric | Train SFT | Train RL | Test | Total |")
    lines.append("|--------|-----------|----------|------|-------|")

    # Count row
    sft_count = stats["train_sft"]["count"]
    rl_count = stats["train_rl"]["count"]
    test_count = stats["test"]["count"]
    lines.append(
        f"| Count | {sft_count:,} | {rl_count:,} | {test_count:,} | {total_puzzles:,} |"
    )

    # Percentage row
    sft_pct = (sft_count / total_puzzles * 100) if total_puzzles > 0 else 0
    rl_pct = (rl_count / total_puzzles * 100) if total_puzzles > 0 else 0
    test_pct = (test_count / total_puzzles * 100) if total_puzzles > 0 else 0
    lines.append(
        f"| Percentage | {sft_pct:.1f}% | {rl_pct:.1f}% | {test_pct:.1f}% | 100.0% |"
    )

    # Average difficulty
    avg_diff_sft = stats["train_sft"]["avg_difficulty"]
    avg_diff_rl = stats["train_rl"]["avg_difficulty"]
    avg_diff_test = stats["test"]["avg_difficulty"]
    avg_diff_total = (
        sum(
            p["difficulty"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    lines.append(
        f"| Avg Difficulty | {avg_diff_sft:.2f} | {avg_diff_rl:.2f} | {avg_diff_test:.2f} | {avg_diff_total:.2f} |"
    )

    # Average quality
    avg_qual_sft = stats["train_sft"]["avg_quality"]
    avg_qual_rl = stats["train_rl"]["avg_quality"]
    avg_qual_test = stats["test"]["avg_quality"]
    avg_qual_total = (
        sum(
            p["quality"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    lines.append(
        f"| Avg Quality | {avg_qual_sft:.2f} | {avg_qual_rl:.2f} | {avg_qual_test:.2f} | {avg_qual_total:.2f} |"
    )

    # Unique creators
    unique_creators_sft = len(stats["train_sft"]["creators"])
    unique_creators_rl = len(stats["train_rl"]["creators"])
    unique_creators_test = len(stats["test"]["creators"])
    unique_creators_total = len(
        set().union(*[stats[s]["creators"] for s in ["train_sft", "train_rl", "test"]])
    )
    lines.append(
        f"| Unique Creators | {unique_creators_sft:,} | {unique_creators_rl:,} | {unique_creators_test:,} | {unique_creators_total:,} |"
    )

    lines.append("")
    lines.append("### Difficulty Bin Breakdown")
    lines.append("")
    lines.append("| Bin | Train SFT | Train RL | Test | Total |")
    lines.append("|-----|-----------|----------|------|-------|")

    all_bins = set()
    for split_stats in stats.values():
        all_bins.update(split_stats["difficulty_bins"].keys())

    for bin_key in sorted(all_bins):
        sft_bin = stats["train_sft"]["difficulty_bins"].get(bin_key, 0)
        rl_bin = stats["train_rl"]["difficulty_bins"].get(bin_key, 0)
        test_bin = stats["test"]["difficulty_bins"].get(bin_key, 0)
        total_bin = sft_bin + rl_bin + test_bin
        lines.append(
            f"| {bin_key} | {sft_bin:,} | {rl_bin:,} | {test_bin:,} | {total_bin:,} |"
        )

    lines.append("")
    lines.append("### Top Countries")
    lines.append("")
    lines.append("| Country | Train SFT | Train RL | Test | Total |")
    lines.append("|---------|-----------|----------|------|-------|")

    all_countries = Counter()
    for split_stats in stats.values():
        all_countries.update(split_stats["countries"])

    for country, total_count in all_countries.most_common(5):
        sft_country = stats["train_sft"]["countries"].get(country, 0)
        rl_country = stats["train_rl"]["countries"].get(country, 0)
        test_country = stats["test"]["countries"].get(country, 0)
        lines.append(
            f"| {country} | {sft_country:,} | {rl_country:,} | {test_country:,} | {total_count:,} |"
        )

    return "\n".join(lines)


def generate_readme(
    splits,
    stats,
    total_puzzles,
    dataset_dict=None,
    template_path="README.template.md",
):
    """Generate README.md from template with statistics."""
    template_file = Path(__file__).parent / template_path

    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    with open(template_file, "r") as f:
        template = f.read()

    # Calculate values for template
    sft_count = stats["train_sft"]["count"]
    rl_count = stats["train_rl"]["count"]
    test_count = stats["test"]["count"]
    sft_pct = (sft_count / total_puzzles * 100) if total_puzzles > 0 else 0
    rl_pct = (rl_count / total_puzzles * 100) if total_puzzles > 0 else 0
    test_pct = (test_count / total_puzzles * 100) if total_puzzles > 0 else 0

    avg_difficulty = (
        sum(
            p["difficulty"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    avg_quality = (
        sum(
            p["quality"]
            for p in splits["train_sft"] + splits["train_rl"] + splits["test"]
        )
        / total_puzzles
        if total_puzzles > 0
        else 0
    )
    unique_creators = len(
        set().union(*[stats[s]["creators"] for s in ["train_sft", "train_rl", "test"]])
    )

    # Get byte sizes from dataset_dict if available
    # Sizes should be pre-calculated and stored in dataset_dict._sizes
    if dataset_dict is not None and hasattr(dataset_dict, "_sizes"):
        # Use pre-calculated sizes from Parquet files
        sizes = dataset_dict._sizes
        sft_num_bytes = sizes["train_sft"]
        rl_num_bytes = sizes["train_rl"]
        test_num_bytes = sizes["test"]
        dataset_size = sizes["dataset_size"]
        download_size = sizes["download_size"]
    elif dataset_dict is not None:
        # Fallback: try to calculate from dataset
        try:
            # Estimate sizes from Arrow in-memory size
            sft_num_bytes = dataset_dict["train_sft"].data.nbytes
            rl_num_bytes = dataset_dict["train_rl"].data.nbytes
            test_num_bytes = dataset_dict["test"].data.nbytes
            dataset_size = sft_num_bytes + rl_num_bytes + test_num_bytes
            download_size = int(dataset_size * 0.9)
        except Exception:
            # Final fallback: rough estimate from JSON
            sft_num_bytes = len(json.dumps(splits["train_sft"]).encode("utf-8"))
            rl_num_bytes = len(json.dumps(splits["train_rl"]).encode("utf-8"))
            test_num_bytes = len(json.dumps(splits["test"]).encode("utf-8"))
            dataset_size = sft_num_bytes + rl_num_bytes + test_num_bytes
            download_size = int(dataset_size * 0.9)
    else:
        # Fallback: estimate sizes (rough approximation)
        # Average puzzle size is roughly 1KB, so estimate from counts
        sft_num_bytes = int(sft_count * 1000)
        rl_num_bytes = int(rl_count * 1000)
        test_num_bytes = int(test_count * 1000)
        dataset_size = sft_num_bytes + rl_num_bytes + test_num_bytes
        download_size = int(dataset_size * 0.9)

    # Generate stats table
    stats_table = generate_stats_table_markdown(splits, stats, total_puzzles)

    # Format template
    readme_content = template.format(
        total_puzzles=total_puzzles,
        sft_count=sft_count,
        rl_count=rl_count,
        test_count=test_count,
        sft_pct=sft_pct,
        rl_pct=rl_pct,
        test_pct=test_pct,
        avg_difficulty=avg_difficulty,
        avg_quality=avg_quality,
        unique_creators=unique_creators,
        stats_table=stats_table,
        sft_num_bytes=sft_num_bytes,
        rl_num_bytes=rl_num_bytes,
        test_num_bytes=test_num_bytes,
        dataset_size=dataset_size,
        download_size=download_size,
    )

    return readme_content


def upload_to_huggingface(repo_name, splits, stats, total_puzzles, commit_message):
    """
    Upload dataset to Hugging Face Hub.

    Args:
        repo_name: e.g., "your-username/connections-puzzles"
        splits: dict with "train_sft", "train_rl", "test" keys containing puzzle lists
        stats: statistics dictionary from calculate_split_stats
        total_puzzles: total number of puzzles
        commit_message: Commit message for the upload
    """
    print("\nLoading splits...")
    train_sft = splits["train_sft"]
    train_rl = splits["train_rl"]
    test_puzzles = splits["test"]

    print(f"Loaded {len(train_sft):,} train_sft puzzles")
    print(f"Loaded {len(train_rl):,} train_rl puzzles")
    print(f"Loaded {len(test_puzzles):,} test puzzles")

    # Create Hugging Face datasets
    print("\nCreating Hugging Face datasets...")
    train_sft_dataset = Dataset.from_list(train_sft)
    train_rl_dataset = Dataset.from_list(train_rl)
    test_dataset = Dataset.from_list(test_puzzles)

    # Create dataset dict with 3-way split
    dataset_dict = DatasetDict(
        {
            "train_sft": train_sft_dataset,
            "train_rl": train_rl_dataset,
            "test": test_dataset,
        }
    )

    print("\nDataset structure:")
    print(f"  Features: {list(train_sft_dataset.features.keys())}")
    print(f"  Train SFT size: {len(train_sft_dataset):,}")
    print(f"  Train RL size: {len(train_rl_dataset):,}")
    print(f"  Test size: {len(test_dataset):,}")

    # Show sample
    print("\nSample puzzle (from train_sft):")
    sample = train_sft_dataset[0]
    for key, value in sample.items():
        if isinstance(value, list) and len(str(value)) > 100:
            print(f"  {key}: [{len(value)} items]")
        else:
            print(f"  {key}: {value}")

    # Upload to Hub
    print(f"\nUploading dataset to {repo_name}...")
    dataset_dict.push_to_hub(repo_name, commit_message=commit_message)

    # Generate and upload README
    print("\nGenerating README.md...")
    readme_content = generate_readme(splits, stats, total_puzzles, dataset_dict)

    # Write README to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    try:
        print("Uploading README.md...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message=f"Update README: {commit_message}",
        )
        print("✅ README.md uploaded successfully!")
    finally:
        # Clean up temp file
        Path(readme_path).unlink()

    print("✅ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_name}")


def main(
    input_file="complete_puzzles_cleaned.jsonl",
    output_dir=".",
    repo_name=None,
    sft_ratio=0.1,
    rl_ratio=0.8,
):
    """
    Main function to generate splits and optionally upload to Hugging Face.

    Args:
        input_file: Path to complete_puzzles_cleaned.jsonl
        output_dir: Directory to write output JSONL files
        repo_name: Hugging Face repo name (e.g., "username/dataset-name")
        sft_ratio: Ratio for train_sft (default 0.1 = 10%)
        rl_ratio: Ratio for train_rl (default 0.8 = 80%)
    """
    print(f"Using hash seed: {HASH_SEED}")
    print(
        f"Split ratios: SFT={sft_ratio * 100:.1f}%, RL={rl_ratio * 100:.1f}%, Test={(1 - sft_ratio - rl_ratio) * 100:.1f}%"
    )

    # Prepare dataset
    puzzles = prepare_connections_dataset(input_file)

    # Create splits
    splits = create_stratified_splits(puzzles, sft_ratio, rl_ratio)

    # Write splits to JSONL files
    write_splits_to_jsonl(splits, output_dir)
    print("\n✅ Splits written to JSONL files")

    # Calculate statistics
    stats = calculate_split_stats(splits)

    # Print statistics table
    print_stats_table(splits, stats, len(puzzles))

    # Ask for upload confirmation
    print("\n" + "=" * 100)
    response = input("Do you want to upload to Hugging Face? (y/n): ").strip().lower()

    if response != "y":
        print("Exiting without uploading.")
        return

    # Prompt for commit message
    commit_message = input("\nEnter commit message (required): ").strip()
    if not commit_message:
        print("No commit message provided. Exiting without uploading.")
        return

    # Upload to Hugging Face
    upload_to_huggingface(repo_name, splits, stats, len(puzzles), commit_message)


if __name__ == "__main__":
    repo_name = "ericbotti/connections-puzzles"

    main(repo_name=repo_name)
