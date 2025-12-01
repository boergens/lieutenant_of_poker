"""
High-level analysis operations for video processing.

This module provides the main entry points for analyzing poker videos,
extracting frames, and generating reports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Callable, List

import cv2

from .frame_extractor import VideoFrameExtractor, FrameInfo
from .game_state import GameStateExtractor, GameState


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
) -> List[GameState]:
    """
    Analyze a video file and extract game states.

    Args:
        video_path: Path to the video file.
        config: Analysis configuration.
        on_progress: Optional callback for progress updates.
        on_debug_frame: Optional callback when a frame needs debugging.
                       Args: (frame_info, state, reason)

    Returns:
        List of GameState objects extracted from the video.
    """
    from .chip_ocr import get_ocr_calls, clear_caches
    from .image_matcher import unmatched_was_saved, reset_unmatched_flag
    from .fast_ocr import set_ocr_debug_context

    extractor = GameStateExtractor()
    clear_caches()

    states = []

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
