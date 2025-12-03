"""
Regression tests for hand history export.

These tests use saved game states with first-frame detection from the video
to test the export pipeline. This ensures the same behavior as the CLI.
"""

import random
from pathlib import Path

import pytest

from lieutenant_of_poker.serialization import load_game_states
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.human_export import export_human
from lieutenant_of_poker.action_log_export import export_action_log
from lieutenant_of_poker.first_frame import detect_from_video


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixed seed for deterministic test output (used by snowie export for simulated cards)
TEST_SEED = 12345


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


def run_export_test(video_num: int, export_func, fixture_suffix: str, normalize_func=None, use_rng=False):
    """Generic export test runner."""
    video_path = FIXTURES_DIR / f"{video_num}.mp4"
    states_path = FIXTURES_DIR / f"video{video_num}_states.json"
    fixture_path = FIXTURES_DIR / f"video{video_num}_export_{fixture_suffix}.txt"

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

    # Load states and generate export
    states = load_game_states(states_path)
    if use_rng:
        rng = random.Random(TEST_SEED)
        actual = export_func(states, button_pos=button_pos, player_names=player_names, rng=rng)
    else:
        actual = export_func(states, button_pos=button_pos, player_names=player_names)

    # Load expected output
    expected = fixture_path.read_text()

    # Normalize if needed
    if normalize_func:
        actual = normalize_func(actual)
        expected = normalize_func(expected)
    else:
        actual = actual.strip()
        expected = expected.strip()

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


class TestExportRegression:
    """Regression tests that compare export output against saved fixtures."""

    # Video 6 tests
    def test_video6_snowie_export(self):
        """Video 6 snowie export matches fixture."""
        run_export_test(6, export_snowie, "snowie", normalize_snowie_export, use_rng=True)

    def test_video6_human_export(self):
        """Video 6 human export matches fixture."""
        run_export_test(6, export_human, "human")

    def test_video6_action_log_export(self):
        """Video 6 action log export matches fixture."""
        run_export_test(6, export_action_log, "action_log")

    # Video 7 tests
    def test_video7_snowie_export(self):
        """Video 7 snowie export matches fixture."""
        run_export_test(7, export_snowie, "snowie", normalize_snowie_export, use_rng=True)

    def test_video7_human_export(self):
        """Video 7 human export matches fixture."""
        run_export_test(7, export_human, "human")

    def test_video7_action_log_export(self):
        """Video 7 action log export matches fixture."""
        run_export_test(7, export_action_log, "action_log")


class TestExportFixturesValid:
    """Tests that export fixtures exist and are valid."""

    # Video 6 fixtures
    def test_video6_states_fixture_exists(self):
        assert (FIXTURES_DIR / "video6_states.json").exists()

    def test_video6_snowie_fixture_exists(self):
        assert (FIXTURES_DIR / "video6_export_snowie.txt").exists()

    def test_video6_human_fixture_exists(self):
        assert (FIXTURES_DIR / "video6_export_human.txt").exists()

    def test_video6_action_log_fixture_exists(self):
        assert (FIXTURES_DIR / "video6_export_action_log.txt").exists()

    # Video 7 fixtures
    def test_video7_states_fixture_exists(self):
        assert (FIXTURES_DIR / "video7_states.json").exists()

    def test_video7_snowie_fixture_exists(self):
        assert (FIXTURES_DIR / "video7_export_snowie.txt").exists()

    def test_video7_human_fixture_exists(self):
        assert (FIXTURES_DIR / "video7_export_human.txt").exists()

    def test_video7_action_log_fixture_exists(self):
        assert (FIXTURES_DIR / "video7_export_action_log.txt").exists()
