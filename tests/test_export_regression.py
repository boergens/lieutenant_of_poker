"""
Regression tests for hand history export.

These tests use saved game states with first-frame detection from the video
to test the snowie export pipeline. This ensures the same behavior as the CLI.
"""

from pathlib import Path
from typing import Optional

import pytest

from lieutenant_of_poker.serialization import load_game_states
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.first_frame import detect_from_video
from lieutenant_of_poker.game_simulator import ShowdownConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Showdown configs extracted from fixture files (deterministic, no RNG)
# force_winner is used for legacy fixtures where opponent cards don't match poker evaluation
SHOWDOWN_CONFIGS = {
    2: ShowdownConfig(
        hand_id="00000000",
        opponent_cards={},  # No showdown - fold winner
    ),
    3: ShowdownConfig(
        hand_id="65917772",
        opponent_cards={"player699561": ["Ks", "Qd"]},  # Victor folded
        force_winner="player699561",
    ),
    4: ShowdownConfig(
        hand_id="65917772",
        opponent_cards={},  # No showdown - fold winner
    ),
    5: ShowdownConfig(
        hand_id="65917772",
        opponent_cards={"Victor": ["5d", "6s"]},
    ),
    6: ShowdownConfig(
        hand_id="65917772",
        opponent_cards={},  # No showdown - fold winner
    ),
    7: ShowdownConfig(
        hand_id="65917772",
        opponent_cards={"Victor": ["5c", "6d"]},
        force_winner="Victor",
    ),
}


def normalize_snowie_export(text: str) -> str:
    """
    Normalize snowie export text for comparison.

    Removes/normalizes lines that change between runs:
    - Timestamps/dates
    """
    lines = []
    for line in text.strip().split("\n"):
        # Skip lines that contain timestamps or dates
        if line.startswith("Date:"):
            continue
        if line.startswith("Time:"):
            continue
        if line.startswith("TimeZone:"):
            continue
        lines.append(line)
    return "\n".join(lines)


def run_snowie_export_test(video_num: int):
    """Run snowie export test for a given video number."""
    video_path = FIXTURES_DIR / f"{video_num}.mp4"
    states_path = FIXTURES_DIR / f"video{video_num}_states.json"
    fixture_path = FIXTURES_DIR / f"video{video_num}_export_snowie.txt"

    if not video_path.exists():
        pytest.skip(f"Video file {video_path} not found")
    if not states_path.exists():
        pytest.skip(f"States file {states_path} not found")
    if not fixture_path.exists():
        pytest.skip(f"Fixture file {fixture_path} not found")

    # Detect first frame info (like CLI does)
    first = detect_from_video(str(video_path))
    button_pos = first.button_index if first.button_index is not None else 0
    player_names = first.player_names

    # Load states and generate export with deterministic showdown config
    states = load_game_states(states_path)
    showdown = SHOWDOWN_CONFIGS.get(video_num)
    actual = export_snowie(states, button_pos=button_pos, player_names=player_names, showdown=showdown)

    # Load expected output
    expected = fixture_path.read_text()

    # Normalize both
    actual = normalize_snowie_export(actual)
    expected = normalize_snowie_export(expected)

    if actual != expected:
        # Show diff for debugging
        actual_lines = actual.split("\n")
        expected_lines = expected.split("\n")

        diff_lines = []
        max_lines = max(len(actual_lines), len(expected_lines))
        for i in range(max_lines):
            actual_line = actual_lines[i] if i < len(actual_lines) else "<missing>"
            expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
            if actual_line != expected_line:
                diff_lines.append(f"Line {i+1}:")
                diff_lines.append(f"  expected: {expected_line}")
                diff_lines.append(f"  actual:   {actual_line}")

        pytest.fail(f"Export mismatch:\n" + "\n".join(diff_lines[:30]))


class TestSnowieExportRegression:
    """Regression tests that compare snowie export output against saved fixtures."""

    def test_video2_snowie_export(self):
        """Video 2 snowie export matches fixture (simple preflop folds)."""
        run_snowie_export_test(2)

    def test_video3_snowie_export(self):
        """Video 3 snowie export matches fixture (all-in call, uncalled bet)."""
        run_snowie_export_test(3)

    def test_video4_snowie_export(self):
        """Video 4 snowie export matches fixture (hero folds to river bet)."""
        run_snowie_export_test(4)

    def test_video5_snowie_export(self):
        """Video 5 snowie export matches fixture."""
        run_snowie_export_test(5)

    def test_video6_snowie_export(self):
        """Video 6 snowie export matches fixture."""
        run_snowie_export_test(6)

    def test_video7_snowie_export(self):
        """Video 7 snowie export matches fixture."""
        run_snowie_export_test(7)


class TestSnowieFixturesExist:
    """Tests that snowie export fixtures exist."""

    def test_video2_fixtures_exist(self):
        assert (FIXTURES_DIR / "video2_states.json").exists()
        assert (FIXTURES_DIR / "video2_export_snowie.txt").exists()

    def test_video3_fixtures_exist(self):
        assert (FIXTURES_DIR / "video3_states.json").exists()
        assert (FIXTURES_DIR / "video3_export_snowie.txt").exists()

    def test_video4_fixtures_exist(self):
        assert (FIXTURES_DIR / "video4_states.json").exists()
        assert (FIXTURES_DIR / "video4_export_snowie.txt").exists()

    def test_video5_fixtures_exist(self):
        assert (FIXTURES_DIR / "video5_states.json").exists()
        assert (FIXTURES_DIR / "video5_export_snowie.txt").exists()

    def test_video6_fixtures_exist(self):
        assert (FIXTURES_DIR / "video6_states.json").exists()
        assert (FIXTURES_DIR / "video6_export_snowie.txt").exists()

    def test_video7_fixtures_exist(self):
        assert (FIXTURES_DIR / "video7_states.json").exists()
        assert (FIXTURES_DIR / "video7_export_snowie.txt").exists()
