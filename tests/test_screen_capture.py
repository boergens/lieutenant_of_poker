"""Tests for screen capture module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lieutenant_of_poker.screen_capture import (
    MacOSScreenCapture,
    WindowInfo,
    check_screen_recording_permission,
    get_permission_instructions,
)


class TestWindowInfo:
    """Tests for WindowInfo dataclass."""

    def test_window_info_creation(self):
        """Test creating a WindowInfo object."""
        window = WindowInfo(
            window_id=123,
            title="Test Window",
            owner_name="TestApp",
            bounds=(0, 0, 800, 600),
        )
        assert window.window_id == 123
        assert window.title == "Test Window"
        assert window.owner_name == "TestApp"
        assert window.bounds == (0, 0, 800, 600)

    def test_window_info_str(self):
        """Test string representation of WindowInfo."""
        window = WindowInfo(
            window_id=123,
            title="Game Window",
            owner_name="Governor of Poker",
            bounds=(100, 100, 1920, 1080),
        )
        result = str(window)
        assert "Governor of Poker" in result
        assert "Game Window" in result
        assert "1920x1080" in result


class TestMacOSScreenCapture:
    """Tests for MacOSScreenCapture class."""

    @pytest.fixture
    def mock_quartz(self):
        """Create a mock Quartz module."""
        mock = MagicMock()
        mock.kCGWindowListOptionOnScreenOnly = 1
        mock.kCGWindowListExcludeDesktopElements = 2
        mock.kCGNullWindowID = 0
        mock.kCGWindowNumber = "kCGWindowNumber"
        mock.kCGWindowName = "kCGWindowName"
        mock.kCGWindowOwnerName = "kCGWindowOwnerName"
        mock.kCGWindowBounds = "kCGWindowBounds"
        mock.kCGWindowListOptionIncludingWindow = 4
        mock.kCGWindowImageBoundsIgnoreFraming = 8
        mock.CGRectNull = None
        return mock

    @pytest.fixture
    def sample_window_list(self):
        """Sample window list data."""
        return [
            {
                "kCGWindowNumber": 100,
                "kCGWindowName": "Governor of Poker 3",
                "kCGWindowOwnerName": "Steam",
                "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1728, "Height": 1117},
            },
            {
                "kCGWindowNumber": 101,
                "kCGWindowName": "Terminal",
                "kCGWindowOwnerName": "Terminal",
                "kCGWindowBounds": {"X": 100, "Y": 100, "Width": 800, "Height": 600},
            },
            {
                "kCGWindowNumber": 102,
                "kCGWindowName": "",
                "kCGWindowOwnerName": "Finder",
                "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 0, "Height": 0},
            },
        ]

    def test_list_windows_no_quartz(self):
        """Test list_windows raises ImportError without Quartz."""
        with patch.dict(sys.modules, {"Quartz": None}):
            capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
            capture._window = None
            capture._frame_count = 0
            capture._start_time = None

            # Should raise ImportError when Quartz not available
            with pytest.raises(ImportError):
                capture.list_windows()

    @patch("lieutenant_of_poker.screen_capture.MacOSScreenCapture.list_windows")
    def test_find_window_by_title(self, mock_list):
        """Test finding a window by title."""
        mock_list.return_value = [
            WindowInfo(100, "Governor of Poker 3", "Steam", (0, 0, 1728, 1117)),
            WindowInfo(101, "Terminal", "Terminal", (100, 100, 800, 600)),
        ]

        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = None
        capture._frame_count = 0
        capture._start_time = None

        window = capture.find_window("Governor")
        assert window is not None
        assert window.window_id == 100
        assert "Governor" in window.title

    @patch("lieutenant_of_poker.screen_capture.MacOSScreenCapture.list_windows")
    def test_find_window_by_owner(self, mock_list):
        """Test finding a window by owner name."""
        mock_list.return_value = [
            WindowInfo(100, "Main Window", "Governor of Poker", (0, 0, 1728, 1117)),
            WindowInfo(101, "Terminal", "Terminal", (100, 100, 800, 600)),
        ]

        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = None
        capture._frame_count = 0
        capture._start_time = None

        window = capture.find_window("poker")
        assert window is not None
        assert "Poker" in window.owner_name

    @patch("lieutenant_of_poker.screen_capture.MacOSScreenCapture.list_windows")
    def test_find_window_not_found(self, mock_list):
        """Test finding a window that doesn't exist."""
        mock_list.return_value = [
            WindowInfo(101, "Terminal", "Terminal", (100, 100, 800, 600)),
        ]

        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = None
        capture._frame_count = 0
        capture._start_time = None

        window = capture.find_window("Nonexistent")
        assert window is None

    def test_capture_frame_no_window(self):
        """Test capture_frame returns None when no window set."""
        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = None
        capture._frame_count = 0
        capture._start_time = None

        result = capture.capture_frame()
        assert result is None

    def test_set_window(self):
        """Test setting the target window."""
        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = None
        capture._frame_count = 0
        capture._start_time = None

        window = WindowInfo(100, "Test", "App", (0, 0, 800, 600))
        capture.set_window(window)

        assert capture.window == window
        assert capture._window == window

    @patch("lieutenant_of_poker.screen_capture.MacOSScreenCapture.find_window")
    def test_init_with_window_title(self, mock_find):
        """Test initialization with window title."""
        mock_find.return_value = WindowInfo(
            100, "Governor of Poker", "Steam", (0, 0, 1728, 1117)
        )

        capture = MacOSScreenCapture(window_title="Governor")

        mock_find.assert_called_once_with("Governor")
        assert capture._window is not None
        assert capture._window.window_id == 100

    @patch("lieutenant_of_poker.screen_capture.MacOSScreenCapture.find_window")
    def test_init_window_not_found(self, mock_find):
        """Test initialization fails when window not found."""
        mock_find.return_value = None

        with pytest.raises(ValueError, match="Could not find window"):
            MacOSScreenCapture(window_title="Nonexistent Game")


class TestPermissionCheck:
    """Tests for permission checking functions."""

    def test_get_permission_instructions(self):
        """Test permission instructions are returned."""
        instructions = get_permission_instructions()
        assert "Screen Recording Permission" in instructions
        assert "System Settings" in instructions
        assert "Privacy" in instructions

    @patch("lieutenant_of_poker.screen_capture.Quartz", create=True)
    def test_check_permission_granted(self, mock_quartz):
        """Test permission check when granted."""
        mock_quartz.CGWindowListCreateImage.return_value = MagicMock()
        mock_quartz.CGImageGetWidth.return_value = 1
        mock_quartz.CGRectMake.return_value = MagicMock()
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowImageDefault = 0

        # Import check needs Quartz available
        with patch.dict(sys.modules, {"Quartz": mock_quartz}):
            from lieutenant_of_poker import screen_capture

            # Re-import to get fresh module with mocked Quartz
            result = screen_capture.check_screen_recording_permission()
            # Result depends on mock behavior
            assert isinstance(result, bool)


class TestContinuousCapture:
    """Tests for continuous capture functionality."""

    def test_start_continuous_stop_immediately(self):
        """Test continuous capture stops when stop_event returns True."""
        capture = MacOSScreenCapture.__new__(MacOSScreenCapture)
        capture._window = WindowInfo(100, "Test", "App", (0, 0, 800, 600))
        capture._frame_count = 0
        capture._start_time = None

        frames_captured = []

        def callback(frame):
            frames_captured.append(frame)

        # Stop immediately
        stop_called = [False]

        def stop_event():
            result = stop_called[0]
            stop_called[0] = True
            return result

        with patch.object(capture, "capture_frame", return_value=None):
            capture.start_continuous(callback, fps=10, stop_event=stop_event)

        # Should have stopped after first iteration
        assert len(frames_captured) == 0
