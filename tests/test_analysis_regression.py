"""
Regression tests for video analysis.

These tests ensure that changes to the analysis module don't break
existing video analysis results.
"""

from pathlib import Path

import pytest

from lieutenant_of_poker.analysis import analyze_video


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def discover_fixtures():
    """Find all fixture numbers that have video files."""
    fixtures = []
    for video in sorted(FIXTURES_DIR.glob("*.mp4")):
        if not video.stem.isdigit():
            continue
        fixtures.append(int(video.stem))
    return fixtures


FIXTURE_NUMS = discover_fixtures()


@pytest.mark.parametrize("num", FIXTURE_NUMS)
def test_video_analysis_succeeds(num):
    """Test that video analysis completes without error."""
    video_path = FIXTURES_DIR / f"{num}.mp4"

    if not video_path.exists():
        pytest.skip(f"Video file {video_path} not found")

    # Run analysis - should not raise
    states = analyze_video(str(video_path))

    # Should produce at least one state
    assert len(states) > 0, f"No states produced for video {num}"
