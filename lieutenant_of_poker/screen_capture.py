"""
Live screen capture for macOS using ScreenCaptureKit.

Provides tools to capture frames from a game window in real-time.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .frame_extractor import FrameInfo


@dataclass
class WindowInfo:
    """Information about a capturable window."""

    window_id: int
    title: str
    owner_name: str
    bounds: tuple[int, int, int, int]  # x, y, width, height

    def __str__(self) -> str:
        return f"{self.owner_name}: {self.title} ({self.bounds[2]}x{self.bounds[3]})"


class ScreenCapture(ABC):
    """Abstract base class for screen capture implementations."""

    @abstractmethod
    def find_window(self, title_pattern: str) -> Optional[WindowInfo]:
        """
        Find a window matching the given title pattern.

        Args:
            title_pattern: Substring to match in window title.

        Returns:
            WindowInfo if found, None otherwise.
        """
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[FrameInfo]:
        """
        Capture a single frame from the target window.

        Returns:
            FrameInfo with captured image, or None if capture failed.
        """
        pass

    @abstractmethod
    def list_windows(self) -> list[WindowInfo]:
        """
        List all available windows.

        Returns:
            List of WindowInfo objects for all windows.
        """
        pass

    def start_continuous(
        self,
        callback: Callable[[FrameInfo], None],
        fps: int = 10,
        stop_event: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Capture frames continuously and call callback for each.

        Args:
            callback: Function to call with each captured frame.
            fps: Target frames per second.
            stop_event: Optional callable that returns True to stop capture.
        """
        interval = 1.0 / fps
        while True:
            if stop_event and stop_event():
                break

            start = time.time()
            frame = self.capture_frame()
            if frame is not None:
                callback(frame)

            # Sleep to maintain target FPS
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class MacOSScreenCapture(ScreenCapture):
    """Screen capture implementation for macOS using Quartz."""

    def __init__(self, window_title: Optional[str] = None):
        """
        Initialize the screen capture.

        Args:
            window_title: Optional window title to capture. If provided,
                         will attempt to find and lock onto this window.
        """
        self._window: Optional[WindowInfo] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

        if window_title:
            self._window = self.find_window(window_title)
            if self._window is None:
                raise ValueError(
                    f"Could not find window with title containing: {window_title}"
                )

    def list_windows(self) -> list[WindowInfo]:
        """List all available windows on macOS."""
        try:
            import Quartz
        except ImportError:
            raise ImportError(
                "Quartz framework not available. "
                "Install with: pip install pyobjc-framework-Quartz"
            )

        windows = []

        # Get list of all windows
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )

        for window in window_list:
            window_id = window.get(Quartz.kCGWindowNumber, 0)
            title = window.get(Quartz.kCGWindowName, "")
            owner = window.get(Quartz.kCGWindowOwnerName, "")
            bounds_dict = window.get(Quartz.kCGWindowBounds, {})

            if bounds_dict:
                bounds = (
                    int(bounds_dict.get("X", 0)),
                    int(bounds_dict.get("Y", 0)),
                    int(bounds_dict.get("Width", 0)),
                    int(bounds_dict.get("Height", 0)),
                )
            else:
                bounds = (0, 0, 0, 0)

            # Skip windows with no size
            if bounds[2] > 0 and bounds[3] > 0:
                windows.append(
                    WindowInfo(
                        window_id=window_id,
                        title=title or "",
                        owner_name=owner or "",
                        bounds=bounds,
                    )
                )

        return windows

    def find_window(self, title_pattern: str) -> Optional[WindowInfo]:
        """Find a window with title containing the pattern."""
        pattern_lower = title_pattern.lower()

        for window in self.list_windows():
            # Check both title and owner name
            if pattern_lower in window.title.lower():
                return window
            if pattern_lower in window.owner_name.lower():
                return window

        return None

    def set_window(self, window: WindowInfo) -> None:
        """Set the target window for capture."""
        self._window = window

    @property
    def window(self) -> Optional[WindowInfo]:
        """Get the current target window."""
        return self._window

    def capture_frame(self) -> Optional[FrameInfo]:
        """Capture a frame from the target window."""
        if self._window is None:
            return None

        try:
            import Quartz
        except ImportError:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        # Capture the window
        image_ref = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,  # Capture full window bounds
            Quartz.kCGWindowListOptionIncludingWindow,
            self._window.window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )

        if image_ref is None:
            return None

        # Get image dimensions
        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)

        if width == 0 or height == 0:
            return None

        # Convert to numpy array
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)

        # Create numpy array from raw data
        np_data = np.frombuffer(data, dtype=np.uint8)
        np_data = np_data.reshape((height, bytes_per_row // 4, 4))

        # Trim to actual width (bytes_per_row might include padding)
        np_data = np_data[:, :width, :]

        # Convert BGRA to BGR (OpenCV format)
        image = np_data[:, :, :3].copy()

        # Calculate timestamp
        timestamp_ms = (time.time() - self._start_time) * 1000

        frame_info = FrameInfo(
            frame_number=self._frame_count,
            timestamp_ms=timestamp_ms,
            image=image,
        )

        self._frame_count += 1
        return frame_info

    def capture_screen_region(
        self, x: int, y: int, width: int, height: int
    ) -> Optional[FrameInfo]:
        """
        Capture a specific region of the screen.

        Args:
            x: Left coordinate.
            y: Top coordinate.
            width: Width of region.
            height: Height of region.

        Returns:
            FrameInfo with captured image, or None if capture failed.
        """
        try:
            import Quartz
        except ImportError:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        # Define capture region
        region = Quartz.CGRectMake(x, y, width, height)

        # Capture the screen region
        image_ref = Quartz.CGWindowListCreateImage(
            region,
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageDefault,
        )

        if image_ref is None:
            return None

        # Get image dimensions
        img_width = Quartz.CGImageGetWidth(image_ref)
        img_height = Quartz.CGImageGetHeight(image_ref)

        if img_width == 0 or img_height == 0:
            return None

        # Convert to numpy array
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)

        np_data = np.frombuffer(data, dtype=np.uint8)
        np_data = np_data.reshape((img_height, bytes_per_row // 4, 4))
        np_data = np_data[:, :img_width, :]

        # Convert BGRA to BGR
        image = np_data[:, :, :3].copy()

        timestamp_ms = (time.time() - self._start_time) * 1000

        frame_info = FrameInfo(
            frame_number=self._frame_count,
            timestamp_ms=timestamp_ms,
            image=image,
        )

        self._frame_count += 1
        return frame_info


def check_screen_recording_permission() -> bool:
    """
    Check if screen recording permission is granted on macOS.

    Returns:
        True if permission is granted, False otherwise.
    """
    try:
        import Quartz

        # Try to capture a small region - will fail without permission
        image_ref = Quartz.CGWindowListCreateImage(
            Quartz.CGRectMake(0, 0, 1, 1),
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageDefault,
        )

        if image_ref is None:
            return False

        # Check if we actually got pixels
        width = Quartz.CGImageGetWidth(image_ref)
        return width > 0

    except Exception:
        return False


def get_permission_instructions() -> str:
    """Get instructions for enabling screen recording permission."""
    return """
Screen Recording Permission Required

To use live screen capture, you need to grant screen recording permission:

1. Open System Settings (or System Preferences on older macOS)
2. Go to Privacy & Security > Privacy > Screen Recording
3. Enable screen recording for Terminal (or your Python IDE)
4. You may need to restart the application after granting permission

If using a virtual environment, make sure to grant permission to the
terminal application running Python, not Python itself.
"""
