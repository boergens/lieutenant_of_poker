"""
Regression tests for hand history export.

These tests use saved game states (not video analysis) to test the export
pipeline independently and quickly. This ensures changes to export code
don't break existing output formats.
"""

from pathlib import Path

import pytest

from lieutenant_of_poker.serialization import load_game_states
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.human_export import export_human


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def normalize_snowie_export(text: str) -> str:
    """
    Normalize snowie export text for comparison.

    Removes/normalizes lines that change between runs:
    - Timestamps/dates
    - GameId (generated from timestamp hash)
    - Showdown lines for opponents (cards are simulated)
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
        # Skip GameId (generated from timestamp hash)
        if line.startswith("GameId:"):
            continue
        # Skip opponent showdown lines (cards are simulated)
        if line.startswith("Showdown:") and "hero" not in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_human_export(text: str) -> str:
    """
    Normalize human export text for comparison.

    Removes/normalizes lines that change between runs:
    - Hand ID (generated from timestamp hash)
    """
    lines = []
    for line in text.strip().split("\n"):
        # Normalize Hand ID line
        if line.startswith("=== Hand #"):
            lines.append("=== Hand #XXXXX ===")
            continue
        lines.append(line)
    return "\n".join(lines)


class TestExportRegression:
    """Regression tests that compare export output against saved fixtures."""

    def test_video5_snowie_export(self):
        """Video 5 snowie export matches fixture."""
        states_path = FIXTURES_DIR / "video5_states.json"
        fixture_path = FIXTURES_DIR / "video5_export_snowie.txt"

        if not states_path.exists():
            pytest.skip(f"States file {states_path} not found")
        if not fixture_path.exists():
            pytest.skip(f"Fixture file {fixture_path} not found")

        # Load states and generate export
        states = load_game_states(states_path)
        actual = export_snowie(states, button_pos=0)

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

    def test_video5_human_export(self):
        """Video 5 human export matches fixture."""
        states_path = FIXTURES_DIR / "video5_states.json"
        fixture_path = FIXTURES_DIR / "video5_export_human.txt"

        if not states_path.exists():
            pytest.skip(f"States file {states_path} not found")
        if not fixture_path.exists():
            pytest.skip(f"Fixture file {fixture_path} not found")

        # Load states and generate export
        states = load_game_states(states_path)
        actual = export_human(states, button_pos=0)

        # Load expected output
        expected = fixture_path.read_text()

        # Compare (normalized to ignore hand ID differences)
        actual_normalized = normalize_human_export(actual)
        expected_normalized = normalize_human_export(expected)

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

            pytest.fail(f"Human export mismatch:\n" + "\n".join(diff_lines[:30]))


class TestExportFixturesValid:
    """Tests that export fixtures exist and are valid."""

    def test_video5_states_fixture_exists(self):
        """Video 5 states fixture exists."""
        assert (FIXTURES_DIR / "video5_states.json").exists()

    def test_video5_snowie_fixture_exists(self):
        """Video 5 snowie export fixture exists."""
        assert (FIXTURES_DIR / "video5_export_snowie.txt").exists()

    def test_video5_human_fixture_exists(self):
        """Video 5 human export fixture exists."""
        assert (FIXTURES_DIR / "video5_export_human.txt").exists()

    def test_video5_snowie_fixture_not_empty(self):
        """Video 5 snowie export fixture has content."""
        content = (FIXTURES_DIR / "video5_export_snowie.txt").read_text()
        assert len(content) > 100
        assert "GameStart" in content

    def test_video5_human_fixture_not_empty(self):
        """Video 5 human export fixture has content."""
        content = (FIXTURES_DIR / "video5_export_human.txt").read_text()
        assert len(content) > 100
        assert "Hand #" in content
