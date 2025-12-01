"""
High-level analysis operations for video processing.

This module provides the main entry points for analyzing poker videos,
extracting frames, and generating reports.
"""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional, Callable, List, TypeVar

import cv2

from .frame_extractor import VideoFrameExtractor, FrameInfo
from .game_state import GameStateExtractor, GameState, PlayerPosition, PlayerState

T = TypeVar("T")


def majority_vote(values: List[T]) -> Optional[T]:
    """
    Return the most common value from a list, or None if the list is empty.

    For values that appear equally often, returns the first one encountered.
    None values are filtered out before voting.
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    counter = Counter(valid_values)
    return counter.most_common(1)[0][0]


def compute_initial_state(states: List[GameState]) -> GameState:
    """
    Compute a consolidated initial state from multiple frame states using majority voting.

    Pools hero cards and chip values from the provided states and returns
    a new GameState with the majority-voted values.

    Args:
        states: List of GameState objects from the first few frames.

    Returns:
        A consolidated GameState with majority-voted values.
    """
    if not states:
        return GameState()

    # Use the first state as the base (for metadata like frame_number, timestamp)
    base_state = states[0]

    # Majority vote for hero cards
    # Cards are compared by their (rank, suit) tuple for hashability
    hero_cards_left = []
    hero_cards_right = []
    for state in states:
        if len(state.hero_cards) >= 1:
            hero_cards_left.append((state.hero_cards[0].rank, state.hero_cards[0].suit))
        if len(state.hero_cards) >= 2:
            hero_cards_right.append((state.hero_cards[1].rank, state.hero_cards[1].suit))

    from .card_detector import Card
    voted_hero_cards = []
    voted_left = majority_vote(hero_cards_left)
    if voted_left:
        voted_hero_cards.append(Card(rank=voted_left[0], suit=voted_left[1]))
    voted_right = majority_vote(hero_cards_right)
    if voted_right:
        voted_hero_cards.append(Card(rank=voted_right[0], suit=voted_right[1]))

    # Majority vote for pot
    pots = [s.pot for s in states]
    voted_pot = majority_vote(pots)

    # Majority vote for hero_chips
    hero_chips_list = [s.hero_chips for s in states]
    voted_hero_chips = majority_vote(hero_chips_list)

    # Majority vote for each player's chips
    # First, collect all player positions that appear
    all_positions = set()
    for state in states:
        all_positions.update(state.players.keys())

    voted_players = {}
    for pos in all_positions:
        chips_list = [s.players.get(pos, PlayerState(position=pos)).chips for s in states]
        voted_chips = majority_vote(chips_list)

        # Get a base PlayerState from the first state that has this position
        base_player = None
        for state in states:
            if pos in state.players:
                base_player = state.players[pos]
                break

        if base_player:
            voted_players[pos] = PlayerState(
                position=pos,
                name=base_player.name,
                chips=voted_chips,
                cards=base_player.cards,
                last_action=base_player.last_action,
                is_active=base_player.is_active,
                is_dealer=base_player.is_dealer,
            )

    return GameState(
        community_cards=[],  # No community cards at start
        hero_cards=voted_hero_cards,
        pot=voted_pot,
        hero_chips=voted_hero_chips,
        players=voted_players,
        street=base_state.street,
        frame_number=base_state.frame_number,
        timestamp_ms=base_state.timestamp_ms,
    )


@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""

    interval_ms: int = 1000
    start_ms: float = 0
    end_ms: Optional[float] = None
    debug_dir: Optional[Path] = None


@dataclass
class AnalysisProgress:
    """Progress information during analysis."""

    current_frame: int
    total_frames: int
    timestamp_ms: float
    ocr_calls: int = 0


def analyze_video(
    video_path: str,
    config: AnalysisConfig,
    on_progress: Optional[Callable[[AnalysisProgress], None]] = None,
    on_debug_frame: Optional[Callable[[FrameInfo, GameState, str], None]] = None,
    initial_frames: int = 3,
) -> List[GameState]:
    """
    Analyze a video file and extract game states.

    Args:
        video_path: Path to the video file.
        config: Analysis configuration.
        on_progress: Optional callback for progress updates.
        on_debug_frame: Optional callback when a frame needs debugging.
                       Args: (frame_info, state, reason)
        initial_frames: Number of frames to pool for initial state (default 3).

    Returns:
        List of GameState objects extracted from the video.
    """
    from .chip_ocr import get_ocr_calls, clear_caches
    from .image_matcher import unmatched_was_saved, reset_unmatched_flag
    from .fast_ocr import set_ocr_debug_context

    extractor = GameStateExtractor()
    clear_caches()

    states = []
    initial_states = []  # Collect first N frames for majority voting

    with VideoFrameExtractor(video_path) as video:
        start_ms = config.start_ms
        end_ms = config.end_ms if config.end_ms else video.duration_seconds * 1000

        total_frames = int((end_ms - start_ms) / config.interval_ms) + 1
        current_frame = 0

        for frame_info in video.iterate_at_interval(
            config.interval_ms, start_ms, end_ms if config.end_ms else None
        ):
            # Reset debug flag before processing
            if on_debug_frame:
                reset_unmatched_flag()

            # Set OCR debug context for this frame
            set_ocr_debug_context(video_path, frame_info.timestamp_ms)

            state = extractor.extract(
                frame_info.image,
                frame_number=frame_info.frame_number,
                timestamp_ms=frame_info.timestamp_ms,
            )

            # Pool first N frames for majority voting
            if current_frame < initial_frames:
                initial_states.append(state)
                # When we have enough frames, compute consolidated initial state
                if len(initial_states) == initial_frames:
                    consolidated = compute_initial_state(initial_states)
                    states.append(consolidated)
            else:
                states.append(state)

            # Check if debug callback should be invoked
            if on_debug_frame:
                has_failure = (
                    state.pot is None
                    or state.hero_chips is None
                    or any(p.chips is None for p in state.players.values())
                )
                if unmatched_was_saved():
                    on_debug_frame(frame_info, state, "unmatched_saved")
                elif has_failure:
                    on_debug_frame(frame_info, state, "detection_failed")

            current_frame += 1

            if on_progress:
                on_progress(
                    AnalysisProgress(
                        current_frame=current_frame,
                        total_frames=total_frames,
                        timestamp_ms=frame_info.timestamp_ms,
                        ocr_calls=get_ocr_calls(),
                    )
                )

        # Handle edge case: video has fewer frames than initial_frames
        if initial_states and not states:
            consolidated = compute_initial_state(initial_states)
            states.append(consolidated)

    return states


def extract_frames(
    video_path: str,
    output_dir: Path,
    interval_ms: int = 1000,
    format: str = "jpg",
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Extract frames from a video file to disk.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save frames.
        interval_ms: Interval between frames in milliseconds.
        format: Output format ('jpg' or 'png').
        start_ms: Start timestamp in milliseconds.
        end_ms: End timestamp in milliseconds (None = end of video).
        on_progress: Optional callback (current, total).

    Returns:
        Number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with VideoFrameExtractor(video_path) as video:
        actual_end_ms = end_ms if end_ms else video.duration_seconds * 1000
        total_frames = int((actual_end_ms - start_ms) / interval_ms) + 1

        for frame_info in video.iterate_at_interval(
            interval_ms, start_ms, actual_end_ms if end_ms else None
        ):
            timestamp_s = frame_info.timestamp_ms / 1000
            filename = f"frame_{timestamp_s:.2f}s.{format}"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame_info.image)
            count += 1

            if on_progress:
                on_progress(count, total_frames)

    return count


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video metadata.
    """
    with VideoFrameExtractor(video_path) as video:
        return {
            "path": video_path,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "duration_seconds": video.duration_seconds,
            "frame_count": video.frame_count,
        }


def generate_diagnostic_report(
    video_path: str,
    output_path: Path,
    frame_number: Optional[int] = None,
    timestamp_s: Optional[float] = None,
) -> dict:
    """
    Generate a diagnostic report for a specific frame.

    Args:
        video_path: Path to the video file.
        output_path: Path for the HTML report.
        frame_number: Frame number to analyze (mutually exclusive with timestamp_s).
        timestamp_s: Timestamp in seconds (mutually exclusive with frame_number).

    Returns:
        Dictionary with report statistics.
    """
    from .diagnostic import DiagnosticExtractor, generate_html_report

    with VideoFrameExtractor(video_path) as video:
        # Determine which frame to analyze
        if frame_number is not None:
            frame_info = video.get_frame_at(frame_number)
        elif timestamp_s is not None:
            frame_info = video.get_frame_at_timestamp(timestamp_s * 1000)
        else:
            frame_info = video.get_frame_at(0)

        if frame_info is None:
            raise ValueError("Could not read frame")

        # Run diagnostic extraction
        extractor = DiagnosticExtractor()
        report = extractor.extract_with_diagnostics(
            frame_info.image,
            frame_number=frame_info.frame_number,
            timestamp_ms=frame_info.timestamp_ms,
        )

        # Generate HTML report
        generate_html_report(report, output_path)

        # Return statistics
        successes = sum(1 for s in report.steps if s.success)
        failures = sum(1 for s in report.steps if not s.success)

        return {
            "frame_number": frame_info.frame_number,
            "timestamp_ms": frame_info.timestamp_ms,
            "steps_succeeded": successes,
            "steps_failed": failures,
            "output_path": output_path,
        }
