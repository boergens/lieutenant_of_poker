"""
Regression tests for hand history export.

Tests auto-discover fixtures in tests/fixtures/ directory.
Each fixture consists of:
  - {n}.mp4 - video file
  - video{n}_states.json - saved game states
  - video{n}_export_snowie.txt - expected snowie output
  - video{n}_config.json (optional) - showdown config for deterministic output
"""

import json
from pathlib import Path

import pytest

from lieutenant_of_poker.serialization import load_game_states
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.first_frame import detect_from_video
from lieutenant_of_poker.game_simulator import ShowdownConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def discover_fixtures():
    """Find all fixture numbers that have complete test files."""
    fixtures = []
    for video in sorted(FIXTURES_DIR.glob("*.mp4")):
        if not video.stem.isdigit():
            continue
        num = int(video.stem)
        states = FIXTURES_DIR / f"video{num}_states.json"
        snowie = FIXTURES_DIR / f"video{num}_export_snowie.txt"
        if states.exists() and snowie.exists():
            fixtures.append(num)
    return fixtures


def load_showdown_config(num: int):
    """Load showdown config from JSON file if it exists."""
    config_path = FIXTURES_DIR / f"video{num}_config.json"
    if not config_path.exists():
        return None, None, None

    data = json.loads(config_path.read_text())
    showdown = ShowdownConfig(
        opponent_cards=data.get("opponent_cards", {}),
        force_winner=data.get("force_winner"),
    )
    hand_id = data.get("hand_id")
    table_background = data.get("table_background")
    return showdown, hand_id, table_background


def normalize_snowie_export(text: str) -> str:
    """Normalize snowie export text for comparison (remove timestamps)."""
    lines = []
    for line in text.strip().split("\n"):
        if line.startswith(("Date:", "Time:", "TimeZone:")):
            continue
        lines.append(line)
    return "\n".join(lines)


FIXTURE_NUMS = discover_fixtures()


@pytest.mark.parametrize("num", FIXTURE_NUMS)
def test_snowie_export(num):
    """Test snowie export matches saved fixture."""
    video_path = FIXTURES_DIR / f"{num}.mp4"
    states_path = FIXTURES_DIR / f"video{num}_states.json"
    fixture_path = FIXTURES_DIR / f"video{num}_export_snowie.txt"

    # Detect first frame info
    first = detect_from_video(str(video_path))
    button_pos = first.button_index if first.button_index is not None else 0
    player_names = first.player_names

    # Load states and config
    states = load_game_states(states_path)
    showdown, hand_id, _table_background = load_showdown_config(num)

    # Generate and compare
    actual = export_snowie(states, button_pos=button_pos, player_names=player_names,
                           showdown=showdown, hand_id=hand_id)
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
