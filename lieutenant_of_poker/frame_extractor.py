"""
Frame extraction utilities for video analysis.

Provides tools to load video files and extract frames at configurable intervals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FrameInfo:
    """Information about an extracted frame."""

    frame_number: int
    timestamp_ms: float
    image: np.ndarray

    @property
    def timestamp_seconds(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp_ms / 1000.0


class VideoFrameExtractor:
    """Extract frames from video files using OpenCV."""

    def __init__(self, video_path: str | Path):
        """
        Initialize the frame extractor.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If the video cannot be opened.
        """
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

    @property
    def fps(self) -> float:
        """Get the video frame rate."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Get total number of frames in the video."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_seconds(self) -> float:
        """Get video duration in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def width(self) -> int:
        """Get video frame width."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get video frame height."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def current_frame(self) -> int:
        """Get current frame position."""
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def current_timestamp_ms(self) -> float:
        """Get current timestamp in milliseconds."""
        return self._cap.get(cv2.CAP_PROP_POS_MSEC)

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame number.

        Args:
            frame_number: Frame number to seek to (0-indexed).

        Returns:
            True if seek was successful, False otherwise.
        """
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def seek_to_timestamp(self, timestamp_ms: float) -> bool:
        """
        Seek to a specific timestamp.

        Args:
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            True if seek was successful, False otherwise.
        """
        return self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

    def read_frame(self) -> Optional[FrameInfo]:
        """
        Read the next frame from the video.

        Returns:
            FrameInfo with frame data, or None if end of video.
        """
        frame_num = self.current_frame

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Get timestamp AFTER read - more accurate after seeks
        timestamp = self.current_timestamp_ms

        return FrameInfo(
            frame_number=frame_num,
            timestamp_ms=timestamp,
            image=frame
        )

    def get_frame_at(self, frame_number: int) -> Optional[FrameInfo]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame number to retrieve (0-indexed).

        Returns:
            FrameInfo with frame data, or None if frame doesn't exist.
        """
        if not self.seek_to_frame(frame_number):
            return None
        return self.read_frame()

    def get_frame_at_timestamp(self, timestamp_ms: float) -> Optional[FrameInfo]:
        """
        Get frame at a specific timestamp.

        Args:
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            FrameInfo with frame data, or None if timestamp is invalid.
        """
        if not self.seek_to_timestamp(timestamp_ms):
            return None
        return self.read_frame()

    def iterate_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Iterator[FrameInfo]:
        """
        Iterate through frames in the video.

        Args:
            start_frame: Frame number to start from (default: 0).
            end_frame: Frame number to stop at (exclusive, default: end of video).
            step: Number of frames to skip between yields (default: 1).

        Yields:
            FrameInfo for each frame in the specified range.
        """
        if end_frame is None:
            end_frame = self.frame_count

        self.seek_to_frame(start_frame)

        current = start_frame
        frames_since_yield = 0
        while current < end_frame:
            frame_info = self.read_frame()
            if frame_info is None:
                break

            # Only yield every step-th frame (sequential read is faster than seeking)
            if frames_since_yield == 0:
                yield frame_info

            frames_since_yield = (frames_since_yield + 1) % step
            current += 1

    def iterate_at_interval(
        self,
        interval_ms: float,
        start_ms: float = 0,
        end_ms: Optional[float] = None
    ) -> Iterator[FrameInfo]:
        """
        Iterate through frames at a fixed time interval.

        Args:
            interval_ms: Time interval between frames in milliseconds.
            start_ms: Start timestamp in milliseconds (default: 0).
            end_ms: End timestamp in milliseconds (default: end of video).

        Yields:
            FrameInfo for each frame at the specified intervals.
        """
        if end_ms is None:
            end_ms = self.duration_seconds * 1000

        current_ms = start_ms
        while current_ms < end_ms:
            frame_info = self.get_frame_at_timestamp(current_ms)
            if frame_info is None:
                break

            yield frame_info
            current_ms += interval_ms

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoFrameExtractor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


def extract_frame_to_file(
    video_path: str | Path,
    output_path: str | Path,
    timestamp_ms: Optional[float] = None,
    frame_number: Optional[int] = None
) -> bool:
    """
    Extract a single frame from a video and save it to a file.

    Args:
        video_path: Path to the video file.
        output_path: Path to save the extracted frame.
        timestamp_ms: Timestamp in milliseconds (mutually exclusive with frame_number).
        frame_number: Frame number to extract (mutually exclusive with timestamp_ms).

    Returns:
        True if extraction was successful, False otherwise.

    Raises:
        ValueError: If neither or both timestamp_ms and frame_number are specified.
    """
    if (timestamp_ms is None) == (frame_number is None):
        raise ValueError("Specify exactly one of timestamp_ms or frame_number")

    with VideoFrameExtractor(video_path) as extractor:
        if timestamp_ms is not None:
            frame_info = extractor.get_frame_at_timestamp(timestamp_ms)
        else:
            frame_info = extractor.get_frame_at(frame_number)

        if frame_info is None:
            return False

        return cv2.imwrite(str(output_path), frame_info.image)
