#!/usr/bin/env uv run
"""
Collate and validate final dataset from token-reduced good and doctored examples

Args:
- good_examples_file: Path to good_examples_reduced.jsonl
- doctored_gameplay_file: Path to doctored_gameplay_reduced.jsonl
- output_file: Path to output final dataset
- --view: Optional flag to start a web server and view results in browser

Combines files and outputs in final format:
- puzzle_id
- prompt
- completion
- reward
- accuracy (correct_guesses / all_guesses)
- guess_history
- categories (preserved from info.categories for viewing)
- doctored (boolean flag indicating if example was doctored)
- doctoring_type (none or gameplay)
- complete_reason (preserved for filtering)

NOTE: Token reduction and validation is done in step_5_reduce_tokens.py,
so all examples here are already within token limits.
"""

import argparse
import json
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

# No shared utility imports needed for this step


def calculate_accuracy(guess_history: list) -> float:
    """
    Calculate accuracy as correct_guesses / all_guesses.

    Returns 0.0 if no guesses, otherwise the ratio of correct guesses.
    """
    if not guess_history:
        return 0.0

    correct_guesses = sum(1 for guess in guess_history if guess.get("status") == "correct")
    total_guesses = len(guess_history)

    return correct_guesses / total_guesses if total_guesses > 0 else 0.0


def synthesize_example(example: Dict[str, Any], doctoring_type: str = "none") -> Dict[str, Any]:
    """
    Convert an example to the final dataset format.

    Args:
        example: Raw example dict
        doctoring_type: "none", "gameplay", or "tokens"

    Fields: puzzle_id, prompt, completion, reward, accuracy, guess_history, categories,
            doctored, doctoring_type, complete_reason
    """
    puzzle_id = example.get("info", {}).get("puzzle_id", "unknown")
    prompt = example.get("prompt", [])
    completion = example.get("completion", [])
    reward = example.get("reward", 0.0)
    guess_history = example.get("guess_history", [])
    categories = example.get("info", {}).get("categories", [])
    complete_reason = example.get("complete_reason", "unknown")

    accuracy = calculate_accuracy(guess_history)

    return {
        "puzzle_id": puzzle_id,
        "prompt": prompt,
        "completion": completion,
        "reward": reward,
        "accuracy": accuracy,
        "guess_history": guess_history,
        "categories": categories,  # Preserve for viewer
        "doctored": doctoring_type != "none",  # Boolean for backward compatibility
        "doctoring_type": doctoring_type,  # "none", "gameplay", or "tokens"
        "complete_reason": complete_reason,  # Preserve for filtering
    }


def load_and_synthesize_examples(
    good_examples_file: Path,
    doctored_gameplay_file: Path,
) -> List[Dict[str, Any]]:
    """
    Load and synthesize examples from token-reduced files.

    Returns:
        final_examples list

    NOTE: No validation needed since all examples are already validated and
    within token limits from step_5_reduce_tokens.py
    """
    if not good_examples_file.exists():
        print(f"Error: Good examples file {good_examples_file} does not exist")
        sys.exit(1)

    print(f"Reading good examples from: {good_examples_file}")
    print(f"Reading doctored gameplay from: {doctored_gameplay_file}")
    print()

    # Read good examples
    good_examples = []
    with open(good_examples_file, "r") as f:
        for line in f:
            if line.strip():
                good_examples.append(json.loads(line))

    print(f"Loaded {len(good_examples)} good examples")

    # Read doctored gameplay examples
    doctored_gameplay = []
    if doctored_gameplay_file.exists():
        with open(doctored_gameplay_file, "r") as f:
            for line in f:
                if line.strip():
                    doctored_gameplay.append(json.loads(line))
        print(f"Loaded {len(doctored_gameplay)} doctored gameplay examples")
    else:
        print(f"Doctored gameplay file not found, skipping")

    # Synthesize (no validation needed - already done in step_5)
    print(f"\nSynthesizing final dataset...")
    final_examples = []

    # Process good examples
    for example in good_examples:
        synthesized = synthesize_example(example, doctoring_type="none")
        final_examples.append(synthesized)

    # Process doctored gameplay examples
    for example in doctored_gameplay:
        synthesized = synthesize_example(example, doctoring_type="gameplay")
        final_examples.append(synthesized)

    print(f"\nTotal examples: {len(final_examples)}")

    return final_examples


def make_handler(html_content: str):
    """Create a handler class with embedded HTML content."""
    class ViewerHandler(BaseHTTPRequestHandler):
        """HTTP handler for serving the viewer with embedded data."""
        
        def do_GET(self):
            """Handle GET requests."""
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        
        def log_message(self, format, *args):
            """Suppress default logging."""
            pass
    
    return ViewerHandler


def create_viewer_html(examples: List[Dict[str, Any]], template_path: Path) -> str:
    """Create HTML with embedded examples data."""
    # Read template
    with open(template_path, "r") as f:
        html_template = f.read()
    
    # Embed examples as JSON in a script tag
    examples_json = json.dumps(examples)
    embedded_script = f"""
    <script>
        // Initialize with embedded data
        initializeWithData({examples_json});
    </script>
    """
    
    # Insert script before closing body tag
    html_content = html_template.replace("</body>", embedded_script + "\n</body>")
    
    return html_content


def start_viewer_server(examples: List[Dict[str, Any]], port: int = 8000):
    """Start a web server to view the synthesized examples."""
    script_dir = Path(__file__).parent
    template_path = script_dir / "rollout_viewer.html"

    if not template_path.exists():
        print(f"Error: Template file {template_path} does not exist")
        sys.exit(1)

    # Create HTML with embedded data
    html_content = create_viewer_html(examples, template_path)

    # Create handler class with embedded HTML
    Handler = make_handler(html_content)

    # Start server
    server_address = ('', port)
    httpd = HTTPServer(server_address, Handler)

    url = f"http://localhost:{port}"
    print(f"\n{'='*60}")
    print("VIEWER SERVER")
    print(f"{'='*60}")
    print(f"Starting server on {url}")
    print("Press Ctrl+C to stop the server")
    print()

    # Open browser
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Warning: Could not open browser automatically: {e}")
        print(f"Please open {url} manually in your browser")

    # Serve forever
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize and validate final dataset from good and doctored examples"
    )
    parser.add_argument(
        "good_examples_file",
        type=Path,
        nargs="?",
        default=Path("good_examples_reduced.jsonl"),
        help="Path to good_examples_reduced.jsonl (default: good_examples_reduced.jsonl in current directory)"
    )
    parser.add_argument(
        "doctored_gameplay_file",
        type=Path,
        nargs="?",
        default=Path("doctored_gameplay_reduced.jsonl"),
        help="Path to doctored_gameplay_reduced.jsonl (default: doctored_gameplay_reduced.jsonl in current directory)"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=Path("sft_examples.jsonl"),
        help="Path to output final dataset (default: sft_examples.jsonl in current directory)"
    )
    parser.add_argument("--view", action="store_true", help="Start web server to view results")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")

    args = parser.parse_args()

    good_examples_file = args.good_examples_file
    doctored_gameplay_file = args.doctored_gameplay_file
    output_file = args.output_file

    # Load and synthesize examples
    final_examples = load_and_synthesize_examples(
        good_examples_file,
        doctored_gameplay_file,
    )

    # Calculate statistics
    good_count = sum(1 for e in final_examples if e.get("doctoring_type") == "none")
    doctored_gameplay_count = sum(1 for e in final_examples if e.get("doctoring_type") == "gameplay")

    total_reward = sum(e["reward"] for e in final_examples)
    avg_reward = total_reward / len(final_examples) if final_examples else 0.0

    total_accuracy = sum(e["accuracy"] for e in final_examples)
    avg_accuracy = total_accuracy / len(final_examples) if final_examples else 0.0

    perfect_accuracy = sum(1 for e in final_examples if e["accuracy"] == 1.0)

    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Total examples: {len(final_examples)}")
    print(f"  Good examples (undoctored): {good_count}")
    print(f"  Doctored gameplay: {doctored_gameplay_count}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average accuracy: {avg_accuracy:.2%}")
    print(f"  Perfect accuracy (100%): {perfect_accuracy} ({perfect_accuracy/len(final_examples)*100:.1f}%)")
    
    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for example in final_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"\n✓ Successfully synthesized dataset")
    print(f"  Output: {output_file}")
    
    # Start viewer if requested
    if args.view:
        start_viewer_server(final_examples, port=args.port)


if __name__ == "__main__":
    main()

