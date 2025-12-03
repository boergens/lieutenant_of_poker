#!/usr/bin/env python3
"""Add a video file to the test fixture suite.

Usage:
    python scripts/add_fixture.py <video_path> [--number N]

This script:
1. Copies the video to tests/fixtures/{n}.mp4
2. Analyzes it and saves states to video{n}_states.json
3. Exports snowie format to video{n}_export_snowie.txt
4. Creates video{n}_config.json with empty showdown config

The test will auto-discover the new fixture.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig
from lieutenant_of_poker.serialization import save_game_states
from lieutenant_of_poker.first_frame import detect_from_video
from lieutenant_of_poker.snowie_export import export_snowie

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def get_next_number() -> int:
    """Find the next available fixture number."""
    existing = list(FIXTURES_DIR.glob("*.mp4"))
    if not existing:
        return 1
    numbers = [int(p.stem) for p in existing if p.stem.isdigit()]
    return max(numbers) + 1 if numbers else 1


def main():
    parser = argparse.ArgumentParser(description="Add a video to test fixtures")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--number", "-n", type=int, help="Fixture number (auto if not specified)")
    parser.add_argument("--hand-id", "-i", help="Hand ID for export (random if not specified)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    num = args.number or get_next_number()

    # Output paths
    video_dest = FIXTURES_DIR / f"{num}.mp4"
    states_path = FIXTURES_DIR / f"video{num}_states.json"
    snowie_path = FIXTURES_DIR / f"video{num}_export_snowie.txt"
    config_path = FIXTURES_DIR / f"video{num}_config.json"

    if video_dest.exists():
        print(f"Error: {video_dest} already exists", file=sys.stderr)
        sys.exit(1)

    print(f"Adding fixture #{num}")
    print(f"  Source: {video_path}")

    # Copy video
    print("Copying video...", end=" ", flush=True)
    shutil.copy(video_path, video_dest)
    print("OK")

    # Analyze
    print("Analyzing video...", end=" ", flush=True)
    states = analyze_video(str(video_dest), AnalysisConfig())
    print(f"OK ({len(states)} states)")

    # Save states
    print(f"Saving {states_path.name}...", end=" ", flush=True)
    save_game_states(states, states_path)
    print("OK")

    # Detect first frame info
    first = detect_from_video(str(video_dest))
    button_pos = first.button_index if first.button_index is not None else 0
    player_names = first.player_names

    # Export snowie (let it generate random hand_id if not specified)
    print(f"Saving {snowie_path.name}...", end=" ", flush=True)
    output = export_snowie(states, button_pos=button_pos, player_names=player_names,
                           hand_id=args.hand_id)
    snowie_path.write_text(output)
    print("OK")

    # Extract hand_id from generated output
    hand_id = None
    for line in output.split("\n"):
        if line.startswith("GameId:"):
            hand_id = line.split(":")[1]
            break

    # Create config with the actual hand_id used
    config = {
        "hand_id": hand_id,
        "opponent_cards": {},
    }
    print(f"Saving {config_path.name}...", end=" ", flush=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("OK")

    print(f"\nDone! Fixture #{num} added.")
    print("Run: pytest tests/test_export_regression.py -v")


if __name__ == "__main__":
    main()
