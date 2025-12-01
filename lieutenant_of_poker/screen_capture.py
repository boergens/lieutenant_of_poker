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


class ScreenCaptureKitCapture(ScreenCapture):
    """
    Screen capture using ScreenCaptureKit (macOS 12.3+).

    This implementation can capture fullscreen apps that run in their own Space,
    unlike the older CGWindowListCreateImage approach.
    """

    def __init__(
        self,
        window_title: Optional[str] = None,
        capture_display: bool = False,
        display_id: int = 0,
    ):
        """
        Initialize ScreenCaptureKit capture.

        Args:
            window_title: Window title pattern to capture. If None and capture_display
                         is False, will capture the main display.
            capture_display: If True, capture entire display instead of a window.
            display_id: Display index to capture (0 = main display).
        """
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._capture_display = capture_display or window_title is None
        self._display_id = display_id
        self._window_title = window_title

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
            import objc
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

        # Select display
        if self._display_id < len(displays):
            self._sc_display = displays[self._display_id]
        else:
            self._sc_display = displays[0]

        # Find window if requested
        if self._window_title and not self._capture_display:
            pattern_lower = self._window_title.lower()
            for window in windows:
                app_name = window.owningApplication().applicationName() or ""
                title = window.title() or ""
                if pattern_lower in app_name.lower() or pattern_lower in title.lower():
                    self._sc_window = window
                    break

        # Create content filter
        if self._sc_window and not self._capture_display:
            # Capture specific window
            self._content_filter = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(
                self._sc_window
            )
        else:
            # Capture entire display
            self._content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                self._sc_display, []
            )

        # Create stream configuration
        self._stream_config = SCK.SCStreamConfiguration.alloc().init()

        # Configure for best quality
        display_width = self._sc_display.width()
        display_height = self._sc_display.height()
        self._stream_config.setWidth_(display_width)
        self._stream_config.setHeight_(display_height)
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

    def find_window(self, title_pattern: str) -> Optional[WindowInfo]:
        """Find a window matching the pattern."""
        pattern_lower = title_pattern.lower()

        for window in self.list_windows():
            if pattern_lower in window.owner_name.lower():
                return window
            if pattern_lower in window.title.lower():
                return window

        return None

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
        elif self._sc_display:
            return WindowInfo(
                window_id=0,
                title=f"Display {self._display_id}",
                owner_name="Display",
                bounds=(0, 0, self._sc_display.width(), self._sc_display.height()),
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
