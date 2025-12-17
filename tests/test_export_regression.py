"""
Regression tests for hand history export.

Tests auto-discover fixtures in tests/fixtures/ directory.
Each fixture consists of:
  - {n}.mp4 - video file
  - video{n}_export_snowie.txt - expected snowie output
  - video{n}_config.json - optional config with showdown cards
"""

import json
from pathlib import Path

import pytest

from lieutenant_of_poker.export import export_video
from lieutenant_of_poker.game_simulator import ShowdownConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def discover_fixtures():
    """Find all fixture numbers that have complete test files."""
    fixtures = []
    for video in sorted(FIXTURES_DIR.glob("*.mp4")):
        if not video.stem.isdigit():
            continue
        num = int(video.stem)
        snowie = FIXTURES_DIR / f"video{num}_export_snowie.txt"
        if snowie.exists():
            fixtures.append(num)
    return fixtures


def normalize_snowie_export(text: str) -> str:
    """Normalize snowie export text for comparison.

    Removes variable fields like dates, times, and game IDs.
    """
    lines = []
    for line in text.strip().split("\n"):
        if line.startswith(("Date:", "Time:", "TimeZone:", "GameId:")):
            continue
        lines.append(line)
    return "\n".join(lines)


FIXTURE_NUMS = discover_fixtures()


def load_showdown_config(num: int) -> ShowdownConfig | None:
    """Load showdown config from JSON file if it exists."""
    config_path = FIXTURES_DIR / f"video{num}_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    if "showdown" not in config:
        return None
    return ShowdownConfig(
        opponent_cards=config["showdown"].get("opponent_cards", {}),
        force_winner=config["showdown"].get("force_winner"),
    )


@pytest.mark.parametrize("num", FIXTURE_NUMS)
def test_snowie_export(num):
    """Test snowie export matches saved fixture."""
    video_path = FIXTURES_DIR / f"{num}.mp4"
    fixture_path = FIXTURES_DIR / f"video{num}_export_snowie.txt"
    showdown = load_showdown_config(num)

    # Generate export using high-level API
    actual = export_video(str(video_path), "snowie", showdown=showdown)
    assert actual is not None, f"export_video returned None for {video_path}"

    expected = fixture_path.read_text()

    actual = normalize_snowie_export(actual)
    expected = normalize_snowie_export(expected)

    if actual != expected:
        actual_lines = actual.split("\n")
        expected_lines = expected.split("\n")
        diff_lines = []
        for i in range(max(len(actual_lines), len(expected_lines))):
            a = actual_lines[i] if i < len(actual_lines) else "<missing>"
            e = expected_lines[i] if i < len(expected_lines) else "<missing>"
            if a != e:
                diff_lines.append(f"Line {i+1}:")
                diff_lines.append(f"  expected: {e}")
                diff_lines.append(f"  actual:   {a}")
        pytest.fail(f"Export mismatch:\n" + "\n".join(diff_lines[:30]))
