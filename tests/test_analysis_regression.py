"""
Regression tests for video analysis.

These tests ensure that changes to the analysis module don't break
existing video analysis results. Each test compares fresh analysis
against saved "golden" fixtures.
"""

from pathlib import Path

import pytest

from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig
from lieutenant_of_poker.serialization import load_game_states, game_state_to_dict


FIXTURES_DIR = Path(__file__).parent / "fixtures"
VIDEOS_DIR = FIXTURES_DIR  # Videos are stored alongside fixtures


def states_match(actual_states, expected_states, tolerance_ms=100):
    """
    Compare two lists of game states for equivalence.

    Args:
        actual_states: States from fresh analysis
        expected_states: States from fixture
        tolerance_ms: Allowed timestamp drift in milliseconds

    Returns:
        Tuple of (match: bool, differences: list)
    """
    differences = []

    if len(actual_states) != len(expected_states):
        differences.append(
            f"State count mismatch: got {len(actual_states)}, expected {len(expected_states)}"
        )
        return False, differences

    for i, (actual, expected) in enumerate(zip(actual_states, expected_states)):
        # Compare key fields
        if actual.pot != expected.pot:
            differences.append(f"State {i}: pot {actual.pot} != {expected.pot}")

        if actual.hero_chips != expected.hero_chips:
            differences.append(f"State {i}: hero_chips {actual.hero_chips} != {expected.hero_chips}")

        if actual.street != expected.street:
            differences.append(f"State {i}: street {actual.street} != {expected.street}")

        # Compare cards
        actual_hero = [c.short_name for c in actual.hero_cards]
        expected_hero = [c.short_name for c in expected.hero_cards]
        if actual_hero != expected_hero:
            differences.append(f"State {i}: hero_cards {actual_hero} != {expected_hero}")

        actual_community = [c.short_name for c in actual.community_cards]
        expected_community = [c.short_name for c in expected.community_cards]
        if actual_community != expected_community:
            differences.append(f"State {i}: community_cards {actual_community} != {expected_community}")

        # Compare player chips
        for pos in actual.players:
            if pos in expected.players:
                actual_chips = actual.players[pos].chips
                expected_chips = expected.players[pos].chips
                if actual_chips != expected_chips:
                    differences.append(
                        f"State {i}: Player {pos} chips {actual_chips} != {expected_chips}"
                    )

    return len(differences) == 0, differences


class TestAnalysisRegression:
    """Regression tests that compare fresh analysis against saved fixtures."""

    @pytest.fixture(autouse=True)
    def check_videos_exist(self):
        """Skip tests if video files are not available."""
        # Videos might not be present in CI environments
        pass

    def _run_regression(self, video_num):
        """Run regression test for a specific video."""
        video_path = VIDEOS_DIR / f"{video_num}.mp4"
        fixture_path = FIXTURES_DIR / f"video{video_num}_states.json"

        if not video_path.exists():
            pytest.skip(f"Video file {video_path} not found")

        if not fixture_path.exists():
            pytest.skip(f"Fixture file {fixture_path} not found")

        # Load expected states from fixture
        expected_states = load_game_states(fixture_path)

        # Run fresh analysis
        actual_states = analyze_video(str(video_path), AnalysisConfig())

        # Compare
        match, differences = states_match(actual_states, expected_states)

        if not match:
            diff_str = "\n".join(f"  - {d}" for d in differences[:20])
            if len(differences) > 20:
                diff_str += f"\n  ... and {len(differences) - 20} more"
            pytest.fail(f"Analysis regression for video {video_num}:\n{diff_str}")

    def test_video6_regression(self):
        """Video 6 analysis matches fixture."""
        self._run_regression(6)


class TestFixturesValid:
    """Tests that fixtures are valid and loadable."""

    def test_video6_fixture_loads(self):
        """Video 6 fixture loads successfully."""
        fixture_path = FIXTURES_DIR / "video6_states.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")
        states = load_game_states(fixture_path)
        assert len(states) > 0
