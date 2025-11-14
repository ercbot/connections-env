import requests
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def load_existing_ids(output_file):
    """Load existing grid IDs from output file to avoid duplicates"""
    existing_ids = set()
    output_path = Path(output_file)
    
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'id' in data:
                        existing_ids.add(data['id'])
            print(f"Found {len(existing_ids)} existing grids in {output_file}")
        except Exception as e:
            print(f"Warning: Could not read existing file {output_file}: {e}")
    
    return existing_ids


def fetch_grids(
    min_quality=4.0, output_file="grids_metadata.jsonl", delay=1.0, max_pages=None, resume=True
):
    """
    Fetch grids with quality >= min_quality from PuzzGrid API

    Args:
        min_quality: Minimum quality rating (default 4.0)
        output_file: Output JSONL file path
        delay: Delay between requests in seconds (be respectful!)
        max_pages: Maximum pages to fetch (None = fetch all pages)
        resume: If True, skip grids that already exist in output file
    """

    base_url = "https://puzzgrid.com/api/grids"

    # Load existing IDs if resuming
    existing_ids = load_existing_ids(output_file) if resume else set()
    
    # First, get total count
    print(f"Fetching grids with quality >= {min_quality}...")

    initial_params = {"qual": min_quality, "page": 1}
    response = requests.get(base_url, params=initial_params)

    if response.status_code != 200:
        print(f"Error fetching initial data: {response.status_code}")
        return

    data = response.json()
    total_grids = int(data.get("total", 0))
    total_pages = int(data.get("pages", 0))

    # Determine how many pages to actually fetch
    pages_to_fetch = min(max_pages, total_pages) if max_pages else total_pages
    estimated_grids = (
        (total_grids * pages_to_fetch) // total_pages if max_pages else total_grids
    )

    print(f"Found {total_grids} grids across {total_pages} total pages")
    print(f"Will fetch {pages_to_fetch} pages (~{estimated_grids} grids)")
    print(f"Using {delay}s delay between requests to be respectful")

    # Confirm before proceeding (skip confirmation for small batches)
    if not max_pages or max_pages > 10:
        proceed = (
            input(f"Proceed with fetching ~{estimated_grids} grids? (y/n): ")
            .lower()
            .strip()
        )
        if proceed != "y":
            print("Aborted.")
            return

    # Prepare output file
    output_path = Path(output_file)
    fetched_count = 0
    skipped_count = 0

    print(f"Starting fetch at {datetime.now()}")
    print(f"Output file: {output_path}")

    # Open file in append mode if resuming, write mode otherwise
    file_mode = "a" if resume and output_path.exists() else "w"
    
    with open(output_path, file_mode) as f:
        with tqdm(total=pages_to_fetch, desc="Fetching pages", unit="page") as pbar:
            for page in range(1, pages_to_fetch + 1):
                params = {"qual": min_quality, "page": page}

                try:
                    response = requests.get(base_url, params=params)

                    if response.status_code != 200:
                        tqdm.write(f"Error on page {page}: {response.status_code}")
                        continue

                    page_data = response.json()
                    rows = page_data.get("rows", [])
                    new_grids_this_page = 0

                    for grid in rows:
                        grid_id = grid.get('id')
                        if grid_id in existing_ids:
                            skipped_count += 1
                            continue
                            
                        # Write immediately to avoid data loss
                        f.write(json.dumps(grid) + "\n")
                        f.flush()  # Force write to disk
                        
                        existing_ids.add(grid_id)  # Track for future duplicates
                        fetched_count += 1
                        new_grids_this_page += 1

                    pbar.set_postfix({
                        "new": new_grids_this_page, 
                        "total": fetched_count,
                        "skipped": skipped_count
                    })

                    # Be respectful - add delay between requests
                    if page < pages_to_fetch:  # Don't delay after last page
                        time.sleep(delay)

                except Exception as e:
                    tqdm.write(f"Error processing page {page}: {e}")
                    continue

                pbar.update(1)

    print(f"Completed! Fetched {fetched_count} new grids, skipped {skipped_count} duplicates")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # Test with a small batch first (uncomment to test)
    # fetch_grids(min_quality=4.0, delay=0.5, max_pages=3, output_file="test_grids.jsonl")

    # Run full fetch of quality >= 4.0 grids
    fetch_grids(min_quality=4.0, delay=1.0)
