"""Tests for frame extractor module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lieutenant_of_poker.frame_extractor import (
    FrameInfo,
    VideoFrameExtractor,
    extract_frame_to_file,
)


@pytest.fixture
def test_video(tmp_path):
    """Create a simple test video for testing."""
    video_path = tmp_path / "test_video.mp4"

    # Create a simple video with 30 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_size = (640, 480)
    out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    for i in range(30):
        # Create a frame with the frame number visible
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 8, i * 8, i * 8)  # Gradient gray
        cv2.putText(frame, f"Frame {i}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)

    out.release()
    return video_path


class TestFrameInfo:
    """Tests for FrameInfo dataclass."""

    def test_timestamp_seconds(self):
        """Test timestamp conversion from ms to seconds."""
        frame = FrameInfo(frame_number=0, timestamp_ms=1500.0, image=np.zeros((1, 1, 3)))
        assert frame.timestamp_seconds == 1.5


class TestVideoFrameExtractor:
    """Tests for VideoFrameExtractor class."""

    def test_open_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            VideoFrameExtractor("/nonexistent/path/video.mp4")

    def test_properties(self, test_video):
        """Test video property getters."""
        with VideoFrameExtractor(test_video) as extractor:
            assert extractor.fps == 30.0
            assert extractor.frame_count == 30
            assert extractor.width == 640
            assert extractor.height == 480
            assert extractor.duration_seconds == 1.0

    def test_read_frame(self, test_video):
        """Test reading a single frame."""
        with VideoFrameExtractor(test_video) as extractor:
            frame = extractor.read_frame()
            assert frame is not None
            assert frame.frame_number == 0
            assert frame.image.shape == (480, 640, 3)

    def test_seek_to_frame(self, test_video):
        """Test seeking to a specific frame."""
        with VideoFrameExtractor(test_video) as extractor:
            extractor.seek_to_frame(15)
            frame = extractor.read_frame()
            assert frame is not None
            assert frame.frame_number == 15

    def test_get_frame_at(self, test_video):
        """Test getting a specific frame by number."""
        with VideoFrameExtractor(test_video) as extractor:
            frame = extractor.get_frame_at(10)
            assert frame is not None
            assert frame.frame_number == 10

    def test_iterate_frames(self, test_video):
        """Test iterating through frames."""
        with VideoFrameExtractor(test_video) as extractor:
            frames = list(extractor.iterate_frames(start_frame=0, end_frame=5))
            assert len(frames) == 5
            assert frames[0].frame_number == 0
            assert frames[4].frame_number == 4

    def test_iterate_frames_with_step(self, test_video):
        """Test iterating with step > 1."""
        with VideoFrameExtractor(test_video) as extractor:
            frames = list(extractor.iterate_frames(start_frame=0, end_frame=10, step=2))
            assert len(frames) == 5
            frame_numbers = [f.frame_number for f in frames]
            assert frame_numbers == [0, 2, 4, 6, 8]

    def test_context_manager(self, test_video):
        """Test context manager properly closes resources."""
        extractor = VideoFrameExtractor(test_video)
        with extractor:
            assert extractor._cap is not None
        assert extractor._cap is None


class TestExtractFrameToFile:
    """Tests for extract_frame_to_file function."""

    def test_extract_by_frame_number(self, test_video, tmp_path):
        """Test extracting a frame by number."""
        output_path = tmp_path / "extracted.jpg"
        result = extract_frame_to_file(test_video, output_path, frame_number=5)
        assert result is True
        assert output_path.exists()

        # Verify the image can be read
        img = cv2.imread(str(output_path))
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_extract_by_timestamp(self, test_video, tmp_path):
        """Test extracting a frame by timestamp."""
        output_path = tmp_path / "extracted.jpg"
        result = extract_frame_to_file(test_video, output_path, timestamp_ms=500)
        assert result is True
        assert output_path.exists()

    def test_extract_requires_one_parameter(self, test_video, tmp_path):
        """Test that exactly one of timestamp or frame_number is required."""
        output_path = tmp_path / "extracted.jpg"

        # Neither specified
        with pytest.raises(ValueError):
            extract_frame_to_file(test_video, output_path)

        # Both specified
        with pytest.raises(ValueError):
            extract_frame_to_file(test_video, output_path, timestamp_ms=500, frame_number=5)
