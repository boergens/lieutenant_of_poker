"""
Live screen capture for macOS using ScreenCaptureKit.

Provides tools to capture frames from the CoinPoker game window in real-time.
"""

import time
from dataclasses import dataclass
from typing import Optional

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


class ScreenCaptureKitCapture:
    """
    Screen capture using ScreenCaptureKit (macOS 12.3+).

    Automatically finds and captures the CoinPoker table window (not the lobby).
    """

    def __init__(self):
        """
        Initialize ScreenCaptureKit capture.

        Automatically finds and captures the CoinPoker table window.
        """
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # These will be set when we get shareable content
        self._sc_display = None
        self._sc_window = None
        self._content_filter = None
        self._stream_config = None

        # Initialize ScreenCaptureKit
        self._init_screencapturekit()

    def _init_screencapturekit(self) -> None:
        """Initialize ScreenCaptureKit and get content filter."""
        import threading

        try:
            import ScreenCaptureKit as SCK
        except ImportError:
            raise ImportError(
                "ScreenCaptureKit not available. "
                "Install with: pip install pyobjc-framework-ScreenCaptureKit"
            )

        # Use threading event to wait for async callback
        event = threading.Event()
        result = {"content": None, "error": None}

        def completion_handler(content, error):
            result["content"] = content
            result["error"] = error
            event.set()

        # Get shareable content
        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(
            completion_handler
        )

        # Wait for completion (with timeout)
        if not event.wait(timeout=5.0):
            raise RuntimeError("Timeout waiting for ScreenCaptureKit content")

        if result["error"]:
            raise RuntimeError(f"ScreenCaptureKit error: {result['error']}")

        content = result["content"]
        if content is None:
            raise RuntimeError("No shareable content available")

        # Get displays and windows
        displays = content.displays()
        windows = content.windows()

        if not displays:
            raise RuntimeError("No displays available")

        self._sc_display = displays[0]

        # Find CoinPoker table window (not lobby)
        # Must be large enough to be an actual table (not a toolbar/notification)
        MIN_TABLE_WIDTH = 800
        MIN_TABLE_HEIGHT = 600

        for window in windows:
            app = window.owningApplication()
            app_name = app.applicationName() if app else ""
            app_name = app_name or ""
            title = window.title() or ""
            frame = window.frame()
            width = int(frame.size.width)
            height = int(frame.size.height)

            # Skip small windows (toolbars, notifications, etc.)
            if width < MIN_TABLE_WIDTH or height < MIN_TABLE_HEIGHT:
                continue

            # Look for CoinPoker window that's not the lobby
            if title.startswith("CoinPoker") and title != "CoinPoker - Lobby":
                self._sc_window = window
                break
            if app_name.startswith("CoinPoker") and not title.endswith("Lobby"):
                self._sc_window = window
                break

        # Create content filter
        if self._sc_window:
            # Capture specific window
            self._content_filter = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(
                self._sc_window
            )
        else:
            # No CoinPoker window found - leave content_filter as None
            return

        # Create stream configuration
        self._stream_config = SCK.SCStreamConfiguration.alloc().init()

        # Set to window's actual dimensions to capture at native resolution
        window_frame = self._sc_window.frame()
        self._stream_config.setWidth_(int(window_frame.size.width))
        self._stream_config.setHeight_(int(window_frame.size.height))
        self._stream_config.setShowsCursor_(False)  # No cursor in capture
        self._stream_config.setPixelFormat_(0x42475241)  # kCVPixelFormatType_32BGRA

    def list_windows(self) -> list[WindowInfo]:
        """List available windows via ScreenCaptureKit."""
        import threading

        try:
            import ScreenCaptureKit as SCK
        except ImportError:
            return []

        event = threading.Event()
        result = {"content": None}

        def completion_handler(content, error):
            result["content"] = content
            event.set()

        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(
            completion_handler
        )

        if not event.wait(timeout=5.0):
            return []

        content = result["content"]
        if content is None:
            return []

        windows = []
        for sc_window in content.windows():
            app = sc_window.owningApplication()
            app_name = app.applicationName() if app else ""
            title = sc_window.title() or ""
            frame = sc_window.frame()

            windows.append(
                WindowInfo(
                    window_id=sc_window.windowID(),
                    title=title,
                    owner_name=app_name or "",
                    bounds=(
                        int(frame.origin.x),
                        int(frame.origin.y),
                        int(frame.size.width),
                        int(frame.size.height),
                    ),
                )
            )

        return windows

    def capture_frame(self) -> Optional[FrameInfo]:
        """Capture a frame using ScreenCaptureKit."""
        import threading

        try:
            import ScreenCaptureKit as SCK
        except ImportError:
            return None

        if self._content_filter is None:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        event = threading.Event()
        result = {"image": None, "error": None}

        def image_handler(cg_image, error):
            result["image"] = cg_image
            result["error"] = error
            event.set()

        # Capture screenshot
        SCK.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            self._content_filter,
            self._stream_config,
            image_handler,
        )

        # Wait for capture (with timeout)
        if not event.wait(timeout=2.0):
            return None

        cg_image = result["image"]
        if cg_image is None:
            return None

        # Convert CGImage to numpy array
        import Quartz

        width = Quartz.CGImageGetWidth(cg_image)
        height = Quartz.CGImageGetHeight(cg_image)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        bits_per_pixel = Quartz.CGImageGetBitsPerPixel(cg_image)

        if width == 0 or height == 0:
            return None

        data_provider = Quartz.CGImageGetDataProvider(cg_image)
        data = Quartz.CGDataProviderCopyData(data_provider)
        np_data = np.frombuffer(data, dtype=np.uint8)

        # Debug: print actual dimensions on first frame
        if self._frame_count == 0:
            print(f"DEBUG: CGImage {width}x{height}, bytes_per_row={bytes_per_row}, "
                  f"bits_per_pixel={bits_per_pixel}, data_size={len(np_data)}",
                  file=__import__('sys').stderr)

        # bytes_per_row may include padding for memory alignment
        # Reshape to (height, bytes_per_row), then slice out the actual pixels
        np_data = np_data[:height * bytes_per_row].reshape((height, bytes_per_row))

        # Each pixel is bits_per_pixel/8 bytes (typically 4 for BGRA)
        bytes_per_pixel = bits_per_pixel // 8
        np_data = np_data[:, :width * bytes_per_pixel].reshape((height, width, bytes_per_pixel))

        # Convert BGRA to BGR (OpenCV format) - drop alpha channel
        image = np_data[:, :, :3].copy()

        timestamp_ms = (time.time() - self._start_time) * 1000

        frame_info = FrameInfo(
            frame_number=self._frame_count,
            timestamp_ms=timestamp_ms,
            image=image,
        )

        self._frame_count += 1
        return frame_info

    @property
    def window(self) -> Optional[WindowInfo]:
        """Get info about the capture target."""
        if self._sc_window:
            app = self._sc_window.owningApplication()
            app_name = app.applicationName() if app else ""
            title = self._sc_window.title() or ""
            frame = self._sc_window.frame()
            return WindowInfo(
                window_id=self._sc_window.windowID(),
                title=title,
                owner_name=app_name,
                bounds=(
                    int(frame.origin.x),
                    int(frame.origin.y),
                    int(frame.size.width),
                    int(frame.size.height),
                ),
            )
        return None


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
