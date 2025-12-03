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

# Fixed seed for deterministic test output
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


class TestExportRegression:
    """Regression tests that compare export output against saved fixtures."""

    def test_video6_snowie_export(self):
        """Video 6 snowie export matches fixture."""
        video_path = FIXTURES_DIR / "6.mp4"
        states_path = FIXTURES_DIR / "video6_states.json"
        fixture_path = FIXTURES_DIR / "video6_export_snowie.txt"

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

        # Load states and generate export with seeded RNG
        states = load_game_states(states_path)
        rng = random.Random(TEST_SEED)
        actual = export_snowie(states, button_pos=button_pos, player_names=player_names, rng=rng)

        # Load expected output
        expected = fixture_path.read_text()

        # Compare (normalized to ignore timestamp differences)
        actual_normalized = normalize_snowie_export(actual)
        expected_normalized = normalize_snowie_export(expected)

        if actual_normalized != expected_normalized:
            # Show diff for debugging
            actual_lines = actual_normalized.split("\n")
            expected_lines = expected_normalized.split("\n")

            diff_lines = []
            max_lines = max(len(actual_lines), len(expected_lines))
            for i in range(max_lines):
                actual_line = actual_lines[i] if i < len(actual_lines) else "<missing>"
                expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
                if actual_line != expected_line:
                    diff_lines.append(f"Line {i+1}:")
                    diff_lines.append(f"  expected: {expected_line}")
                    diff_lines.append(f"  actual:   {actual_line}")

            pytest.fail(f"Snowie export mismatch:\n" + "\n".join(diff_lines[:30]))

    def test_video6_human_export(self):
        """Video 6 human export matches fixture."""
        video_path = FIXTURES_DIR / "6.mp4"
        states_path = FIXTURES_DIR / "video6_states.json"
        fixture_path = FIXTURES_DIR / "video6_export_human.txt"

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

        # Load states and generate export with seeded RNG
        states = load_game_states(states_path)
        rng = random.Random(TEST_SEED)
        actual = export_human(states, button_pos=button_pos, player_names=player_names, rng=rng)

        # Load expected output
        expected = fixture_path.read_text()

        # Compare
        if actual.strip() != expected.strip():
            # Show diff for debugging
            actual_lines = actual.strip().split("\n")
            expected_lines = expected.strip().split("\n")

            diff_lines = []
            max_lines = max(len(actual_lines), len(expected_lines))
            for i in range(max_lines):
                actual_line = actual_lines[i] if i < len(actual_lines) else "<missing>"
                expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
                if actual_line != expected_line:
                    diff_lines.append(f"Line {i+1}:")
                    diff_lines.append(f"  expected: {expected_line}")
                    diff_lines.append(f"  actual:   {actual_line}")

            pytest.fail(f"Human export mismatch:\n" + "\n".join(diff_lines[:30]))

    def test_video6_action_log_export(self):
        """Video 6 action log export matches fixture."""
        video_path = FIXTURES_DIR / "6.mp4"
        states_path = FIXTURES_DIR / "video6_states.json"
        fixture_path = FIXTURES_DIR / "video6_export_action_log.txt"

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

        # Load states and generate export with seeded RNG
        states = load_game_states(states_path)
        rng = random.Random(TEST_SEED)
        actual = export_action_log(states, button_pos=button_pos, player_names=player_names, rng=rng)

        # Load expected output
        expected = fixture_path.read_text()

        # Compare
        if actual.strip() != expected.strip():
            # Show diff for debugging
            actual_lines = actual.strip().split("\n")
            expected_lines = expected.strip().split("\n")

            diff_lines = []
            max_lines = max(len(actual_lines), len(expected_lines))
            for i in range(max_lines):
                actual_line = actual_lines[i] if i < len(actual_lines) else "<missing>"
                expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
                if actual_line != expected_line:
                    diff_lines.append(f"Line {i+1}:")
                    diff_lines.append(f"  expected: {expected_line}")
                    diff_lines.append(f"  actual:   {actual_line}")

            pytest.fail(f"Action log export mismatch:\n" + "\n".join(diff_lines[:30]))


class TestExportFixturesValid:
    """Tests that export fixtures exist and are valid."""

    def test_video6_states_fixture_exists(self):
        """Video 6 states fixture exists."""
        assert (FIXTURES_DIR / "video6_states.json").exists()

    def test_video6_snowie_fixture_exists(self):
        """Video 6 snowie export fixture exists."""
        assert (FIXTURES_DIR / "video6_export_snowie.txt").exists()

    def test_video6_human_fixture_exists(self):
        """Video 6 human export fixture exists."""
        assert (FIXTURES_DIR / "video6_export_human.txt").exists()

    def test_video6_action_log_fixture_exists(self):
        """Video 6 action log export fixture exists."""
        assert (FIXTURES_DIR / "video6_export_action_log.txt").exists()

    def test_video6_snowie_fixture_not_empty(self):
        """Video 6 snowie export fixture has content."""
        content = (FIXTURES_DIR / "video6_export_snowie.txt").read_text()
        assert len(content) > 100
        assert "GameStart" in content

    def test_video6_human_fixture_not_empty(self):
        """Video 6 human export fixture has content."""
        content = (FIXTURES_DIR / "video6_export_human.txt").read_text()
        assert len(content) > 100
        assert "Hand #" in content

    def test_video6_action_log_fixture_not_empty(self):
        """Video 6 action log export fixture has content."""
        content = (FIXTURES_DIR / "video6_export_action_log.txt").read_text()
        assert len(content) > 10
        assert "posts" in content
