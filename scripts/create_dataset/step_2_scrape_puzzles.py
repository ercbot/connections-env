import requests
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup


def caesar_decode_char(char, shift=-3):
    """
    Apply Caesar cipher shift with proper UTF-16 surrogate handling.

    The website uses JavaScript which internally uses UTF-16 encoding.
    When shifting characters, it shifts UTF-16 code units:
    - BMP characters (≤ U+FFFF): single code unit → simple shift
    - Supplementary plane (> U+FFFF): surrogate pair → each surrogate shifted separately

    This decoder handles both cases uniformly by attempting surrogate decode
    on all characters. If the result would be invalid surrogates, we know it's
    a BMP character and use simple shift instead.
    """
    code = ord(char)

    # Attempt to decode as if this were a shifted UTF-16 surrogate pair
    adjusted = code - 0x10000
    high = (adjusted >> 10) + 0xD800
    low = (adjusted & 0x3FF) + 0xDC00

    # Reverse the shift on each surrogate
    high_orig = high + shift  # shift is -3, so this subtracts
    low_orig = low + shift

    # Check if this produces valid surrogates
    if 0xD800 <= high_orig <= 0xDBFF and 0xDC00 <= low_orig <= 0xDFFF:
        # Valid surrogates → this was a supplementary plane character
        # Recombine surrogates to get original code point
        orig_code = ((high_orig - 0xD800) << 10) + (low_orig - 0xDC00) + 0x10000
        return chr(orig_code)

    # Invalid surrogates → this is a BMP character (single UTF-16 code unit)
    # Just apply simple shift
    return chr(code + shift)


def decode_segment(segment):
    """Decode a segment of the encoded string"""
    decoded = ""
    for char in segment.strip():
        decoded += caesar_decode_char(char)
    return decoded


def get_puzzle_content(grid_id):
    """Extract puzzle content for a specific grid ID"""
    url = f"https://puzzgrid.com/grid/{grid_id}"

    try:
        response = requests.get(url)

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}"}

        # Parse HTML to find the __NEXT_DATA__ script tag
        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})

        if not script_tag:
            return {"error": "No __NEXT_DATA__ found"}

        # Parse the JSON data
        next_data = json.loads(script_tag.string)
        page_props = next_data.get("props", {}).get("pageProps", {})
        groups_encoded = page_props.get("groups_o", "")

        if not groups_encoded:
            return {"error": "No groups_o data found"}

        puzzle_content = {"title": page_props.get("title", ""), "groups": []}

        # Remove leading ^
        if groups_encoded.startswith("^"):
            groups_encoded = groups_encoded[1:]

        # Split by ~% to get each group section
        group_sections = groups_encoded.split("~%")

        for section in group_sections:
            if not section or not section.startswith("wlohv%=^%"):
                continue

            # Remove the prefix
            content = section[9:]

            # Find where the tiles end
            tiles_end = content.find("%`/")
            if tiles_end == -1:
                continue

            tiles_part = content[:tiles_end]

            # Split tiles by %/%
            tile_strings = tiles_part.split("%/%")

            tiles = []
            for tile_str in tile_strings:
                if tile_str:
                    decoded = decode_segment(tile_str)
                    tiles.append(decoded)

            # Parse description and linking terms (order can vary!)
            rest_part = content[tiles_end + 3 :]

            description = ""
            linking_terms = ""

            # Find both fields regardless of order
            if "ghvfulswlrq%=" in rest_part:
                desc_start = rest_part.find("ghvfulswlrq%=") + 14
                # Find the end - look for next %/ or end of string
                desc_end = rest_part.find("%/", desc_start)
                if desc_end == -1:
                    # If no %/ found, go to end of string
                    desc_end = len(rest_part)
                desc_encoded = rest_part[desc_start:desc_end].rstrip("\x80/`},")
                description = decode_segment(desc_encoded)

            if "olqnlqjWhupv%=" in rest_part:
                terms_start = rest_part.find("olqnlqjWhupv%=") + 15
                # Find the end - look for next %/ or end of string  
                terms_end = rest_part.find("%/", terms_start)
                if terms_end == -1:
                    # If no %/ found, go to end of string
                    terms_end = len(rest_part)
                terms_encoded = rest_part[terms_start:terms_end].rstrip("\x80/`")
                linking_terms = decode_segment(terms_encoded)

            group_data = {
                "words": tiles,
                "description": description,
                "linking_terms": linking_terms,
            }

            puzzle_content["groups"].append(group_data)

        return puzzle_content

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


def clean_and_load_existing_puzzles(output_file):
    """Load existing puzzles, clean out null entries, and return successful IDs and failed IDs"""
    existing_ids = set()
    failed_ids = set()
    clean_puzzles = []
    output_path = Path(output_file)

    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if "id" in data:
                        if data.get("puzzle_content") is None and "scrape_error" in data:
                            # This puzzle failed - mark for retry
                            failed_ids.add(data["id"])
                            print(f"Found failed puzzle {data['id']}: {data['scrape_error']}")
                        else:
                            # This puzzle succeeded - keep it
                            existing_ids.add(data["id"])
                            clean_puzzles.append(data)
                            
            # If we found failed puzzles, rewrite the file without them
            if failed_ids:
                print(f"Cleaning {len(failed_ids)} failed puzzles from {output_file}")
                backup_file = output_path.with_suffix('.jsonl.backup')
                
                # Create backup
                output_path.rename(backup_file)
                print(f"Created backup: {backup_file}")
                
                # Write clean version
                with open(output_path, "w") as f:
                    for puzzle in clean_puzzles:
                        f.write(json.dumps(puzzle) + "\n")
                
                print(f"Cleaned file: {len(clean_puzzles)} successful puzzles, {len(failed_ids)} will be retried")
            else:
                print(f"Found {len(existing_ids)} existing puzzles in {output_file} (all successful)")
                
        except Exception as e:
            print(f"Warning: Could not read existing file {output_file}: {e}")
            return set(), set()

    return existing_ids, failed_ids


def scrape_puzzles(
    metadata_file="grids_metadata.jsonl",
    output_file="complete_puzzles.jsonl",
    delay=1.0,
    max_puzzles=None,
    resume=True,
):
    """
    Scrape puzzle content for grids from metadata file

    Args:
        metadata_file: JSONL file with grid metadata from step 1
        output_file: Output JSONL file with complete puzzle data
        delay: Delay between requests in seconds (be respectful!)
        max_puzzles: Maximum puzzles to scrape (None = all)
        resume: If True, skip puzzles that already exist in output file
    """

    # Load metadata
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_file} not found!")
        return

    # Load grid metadata
    grids = []
    with open(metadata_path, "r") as f:
        for line in f:
            grid = json.loads(line.strip())
            grids.append(grid)

    print(f"Loaded {len(grids)} grids from {metadata_file}")

    # Limit if requested
    if max_puzzles:
        grids = grids[:max_puzzles]
        print(f"Limited to first {len(grids)} grids")

    # Load existing puzzle IDs and clean failed ones if resuming
    if resume:
        existing_ids, failed_ids = clean_and_load_existing_puzzles(output_file)
        # Add failed IDs back to the processing queue for retry
        retry_grids = [g for g in grids if g["id"] in failed_ids]
        if retry_grids:
            print(f"Will retry {len(retry_grids)} failed puzzles")
    else:
        existing_ids, failed_ids = set(), set()

    # Filter out successfully processed puzzles (but include failed ones for retry)
    grids_to_process = [g for g in grids if g["id"] not in existing_ids]

    print(
        f"Will process {len(grids_to_process)} grids (skipping {len(grids) - len(grids_to_process)} existing successful)"
    )
    if failed_ids:
        print(f"Including {len(failed_ids)} failed puzzles for retry")
    print(f"Using {delay}s delay between requests to be respectful")

    # Confirm before proceeding
    if not max_puzzles or len(grids_to_process) > 20:
        proceed = (
            input(f"Proceed with scraping {len(grids_to_process)} puzzles? (y/n): ")
            .lower()
            .strip()
        )
        if proceed != "y":
            print("Aborted.")
            return

    # Process puzzles
    output_path = Path(output_file)
    processed_count = 0
    error_count = 0

    print(f"Starting scrape at {datetime.now()}")
    print(f"Output file: {output_path}")

    # Open file in append mode if resuming, write mode otherwise
    file_mode = "a" if resume and output_path.exists() else "w"

    with open(output_path, file_mode) as f:
        with tqdm(
            total=len(grids_to_process), desc="Scraping puzzles", unit="puzzle"
        ) as pbar:
            for grid_metadata in grids_to_process:
                grid_id = grid_metadata["id"]

                try:
                    # Get puzzle content
                    puzzle_content = get_puzzle_content(grid_id)

                    # Combine metadata with puzzle content
                    if "error" in puzzle_content:
                        # Log error but save metadata anyway
                        complete_puzzle = {
                            **grid_metadata,
                            "puzzle_content": None,
                            "scrape_error": puzzle_content["error"],
                            "scraped_at": datetime.now().isoformat(),
                        }
                        error_count += 1
                        tqdm.write(
                            f"Error scraping {grid_id}: {puzzle_content['error']}"
                        )
                    else:
                        # Successful scrape
                        complete_puzzle = {
                            **grid_metadata,
                            "puzzle_content": puzzle_content,
                            "scraped_at": datetime.now().isoformat(),
                        }

                    # Write immediately to avoid data loss
                    f.write(json.dumps(complete_puzzle) + "\n")
                    f.flush()

                    processed_count += 1
                    pbar.set_postfix(
                        {
                            "processed": processed_count,
                            "errors": error_count,
                            "success_rate": f"{((processed_count - error_count) / processed_count * 100):.1f}%",
                        }
                    )

                    # Be respectful - add delay between requests
                    time.sleep(delay)

                except Exception as e:
                    error_count += 1
                    tqdm.write(f"Exception processing {grid_id}: {e}")
                    continue

                pbar.update(1)

    success_count = processed_count - error_count
    print(f"Completed! Processed {processed_count} puzzles")
    print(f"Successful: {success_count}, Errors: {error_count}")
    if processed_count > 0:
        print(f"Success rate: {(success_count / processed_count * 100):.1f}%")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Connections puzzles from PuzzGrid")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch instead of resuming (ignores existing puzzles)"
    )
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=None,
        help="Maximum number of puzzles to scrape (default: all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Run full scrape
    scrape_puzzles(
        resume=not args.no_resume,
        max_puzzles=args.max_puzzles,
        delay=args.delay
    )
