"""
Video recording from frame sources.

Provides a simple VideoRecorder that can record frames from ScreenCaptureKitCapture
to a video file.
"""

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

from .frame_extractor import FrameInfo
from .screen_capture import ScreenCaptureKitCapture

# Path to the mask file (in package directory)
MASK_PATH = Path(__file__).parent / "mask.png"


class BrightnessDetector:
    """
    Detects recording start/stop based on brightness in a masked region.

    Uses a mask image to define the region of interest. When the average
    brightness in that region crosses a threshold for consecutive frames,
    it triggers start/stop events.
    """

    def __init__(
        self,
        mask_path: Path = MASK_PATH,
        threshold: float = 250.0,
        consecutive_frames: int = 3,
    ):
        """
        Initialize the brightness detector.

        Args:
            mask_path: Path to the mask image (grayscale PNG).
            threshold: Brightness threshold (0-255).
            consecutive_frames: Number of consecutive frames needed to trigger.
        """
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self._mask: Optional[np.ndarray] = None
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}  # Cache resized masks
        self._brightness_history: deque = deque(maxlen=consecutive_frames)
        self._is_bright = False

        # Load mask
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Normalize to binary mask (0 or 1)
                self._mask = (mask_img > 127).astype(np.uint8)

    @property
    def is_available(self) -> bool:
        """Check if mask was loaded successfully."""
        return self._mask is not None

    def check_frame(self, frame: FrameInfo) -> Optional[bool]:
        """
        Check a frame for brightness trigger.

        Args:
            frame: The captured frame.

        Returns:
            True if should start recording, False if should stop, None if no change.
        """
        if self._mask is None:
            return None

        image = frame.image
        frame_shape = (image.shape[0], image.shape[1])

        # Get cached mask for this frame size, or resize and cache
        if frame_shape in self._mask_cache:
            mask = self._mask_cache[frame_shape]
        elif self._mask.shape[:2] == frame_shape:
            mask = self._mask
        else:
            mask = cv2.resize(self._mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._mask_cache[frame_shape] = mask

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate average brightness in masked region
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) == 0:
            return None

        avg_brightness = np.mean(masked_pixels)
        self._brightness_history.append(avg_brightness)

        # Need enough history
        if len(self._brightness_history) < self.consecutive_frames:
            return None

        # Check if all recent frames are above/below threshold
        all_bright = all(b > self.threshold for b in self._brightness_history)
        all_dark = all(b <= self.threshold for b in self._brightness_history)

        # Trigger on state change
        if all_bright and not self._is_bright:
            self._is_bright = True
            return True  # Start recording
        elif all_dark and self._is_bright:
            self._is_bright = False
            return False  # Stop recording

        return None


class VideoRecorder:
    """
    Records frames to a video file.

    Can be used standalone or composed with other components.
    Buffers the last few frames so they can be discarded when recording stops
    (useful when stop is triggered by detecting something in those frames).
    """

    def __init__(self, fps: int = 10, codec: str = "mp4v", trailing_frames_to_drop: int = 3):
        """
        Initialize the recorder.

        Args:
            fps: Frames per second for output video.
            codec: FourCC codec code (default: 'mp4v' for .mp4).
            trailing_frames_to_drop: Number of trailing frames to discard on stop.
        """
        self.fps = fps
        self.codec = codec
        self.trailing_frames_to_drop = trailing_frames_to_drop
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[Path] = None
        self._frame_count = 0
        self._frame_buffer: deque = deque(maxlen=trailing_frames_to_drop)

    def start(self, output_path: Union[str, Path]) -> None:
        """Start recording to the specified path."""
        if self._writer is not None:
            self.stop()
        self._recording_path = Path(output_path)
        self._frame_count = 0
        self._frame_buffer.clear()
        # Writer is lazily initialized on first frame (need dimensions)

    def stop(self) -> Optional[Path]:
        """Stop recording and return the output path. Discards buffered trailing frames."""
        # Discard buffered frames (don't write them)
        self._frame_buffer.clear()

        if self._writer is not None:
            self._writer.release()
            self._writer = None
        path = self._recording_path
        self._recording_path = None
        return path

    def write_frame(self, frame: FrameInfo) -> None:
        """Write a frame to the video file (with trailing frame buffer)."""
        if self._recording_path is None:
            return

        image = frame.image

        # Initialize writer on first frame (now we know dimensions)
        if self._writer is None:
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self._recording_path),
                fourcc,
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                print(f"ERROR: Failed to open VideoWriter for {self._recording_path}",
                      file=__import__('sys').stderr)
                print(f"  Dimensions: {width}x{height}, FPS: {self.fps}, Codec: {self.codec}",
                      file=__import__('sys').stderr)

        # Buffer frames - only write when buffer is full (delayed by trailing_frames_to_drop)
        if len(self._frame_buffer) == self.trailing_frames_to_drop:
            # Write the oldest buffered frame
            oldest_image = self._frame_buffer[0]
            self._writer.write(oldest_image)
            self._frame_count += 1

        # Add current frame to buffer
        self._frame_buffer.append(image.copy())

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording_path is not None

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    @property
    def current_path(self) -> Optional[Path]:
        """Path to current recording, if any."""
        return self._recording_path


class RecordingSession:
    """
    A complete recording session with capture and recording.

    Provides a simple interface for the 'just record' use case.
    """

    def __init__(
        self,
        capture: ScreenCaptureKitCapture,
        output_dir: Union[str, Path] = ".",
        fps: int = 10,
        codec: str = "mp4v",
        file_prefix: str = "recording",
        auto_detect: bool = False,
        on_recording_change: Optional[Callable[[bool, Optional[Path]], None]] = None,
        on_frame_drop: Optional[Callable[[float, float], None]] = None,
        frame_drop_threshold: float = 0.75,
        frame_drop_cooldown: float = 5.0,
        debug: bool = False,
        on_debug_stats: Optional[Callable[[dict], None]] = None,
        debug_interval: float = 5.0,
    ):
        """
        Initialize a recording session.

        Args:
            capture: Screen capture implementation to use.
            output_dir: Directory to save recordings.
            fps: Frames per second.
            codec: Video codec.
            file_prefix: Prefix for auto-generated filenames.
            auto_detect: Enable automatic recording based on mask brightness.
            on_recording_change: Callback when recording starts/stops.
                                 Args: (is_recording, path_if_stopped)
            on_frame_drop: Callback when frame drops are detected.
                          Args: (actual_fps, target_fps)
            frame_drop_threshold: FPS ratio below which to trigger notification (0.75 = 75%).
            frame_drop_cooldown: Seconds between frame drop notifications.
            debug: Enable debug timing instrumentation.
            on_debug_stats: Callback for debug stats output.
                           Args: (stats_dict)
            debug_interval: Seconds between debug stats output.
        """
        self.capture = capture
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.file_prefix = file_prefix
        self.on_recording_change = on_recording_change
        self.on_frame_drop = on_frame_drop
        self.frame_drop_threshold = frame_drop_threshold
        self.frame_drop_cooldown = frame_drop_cooldown
        self.debug = debug
        self.on_debug_stats = on_debug_stats
        self.debug_interval = debug_interval

        self._recorder = VideoRecorder(fps=fps, codec=codec)
        self._running = False
        self._recordings: list[Path] = []

        # Frame drop tracking
        self._frame_times: deque = deque(maxlen=30)  # Track last 30 frame times
        self._last_frame_drop_notification: float = 0.0

        # Auto-detection
        self._detector: Optional[BrightnessDetector] = None
        if auto_detect:
            self._detector = BrightnessDetector()
            if not self._detector.is_available:
                self._detector = None

        # Debug timing stats
        self._debug_capture_times: deque = deque(maxlen=100)
        self._debug_detect_times: deque = deque(maxlen=100)
        self._debug_write_times: deque = deque(maxlen=100)
        self._debug_loop_times: deque = deque(maxlen=100)
        self._debug_sleep_drifts: deque = deque(maxlen=100)
        self._debug_last_stats_time: float = 0.0
        self._debug_overruns: int = 0
        self._debug_total_frames: int = 0

    @property
    def auto_detect_available(self) -> bool:
        """Check if auto-detection is available."""
        return self._detector is not None

    def _check_frame_drops(self) -> None:
        """Check for frame drops and notify if needed."""
        if self.on_frame_drop is None or len(self._frame_times) < 10:
            return

        # Calculate actual FPS from recent frame times
        if len(self._frame_times) < 2:
            return

        frame_times = list(self._frame_times)
        total_time = frame_times[-1] - frame_times[0]
        if total_time <= 0:
            return

        actual_fps = (len(frame_times) - 1) / total_time

        # Check if we're dropping frames (below threshold of target FPS)
        fps_ratio = actual_fps / self.fps
        if fps_ratio < self.frame_drop_threshold:
            # Check cooldown
            now = time.time()
            if now - self._last_frame_drop_notification >= self.frame_drop_cooldown:
                self._last_frame_drop_notification = now
                self.on_frame_drop(actual_fps, self.fps)

    def toggle_recording(self) -> tuple[bool, Optional[Path]]:
        """
        Toggle recording on/off.

        Returns:
            Tuple of (is_now_recording, path_if_just_stopped)
        """
        if self._recorder.is_recording:
            path = self._recorder.stop()
            if path:
                self._recordings.append(path)
            return False, path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"{self.file_prefix}_{timestamp}.mp4"
            self._recorder.start(filename)
            return True, None

    def _start_recording(self) -> Path:
        """Start a new recording and return the path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{self.file_prefix}_{timestamp}.mp4"
        self._recorder.start(filename)
        return filename

    def _stop_recording(self) -> Optional[Path]:
        """Stop recording and return the path."""
        path = self._recorder.stop()
        if path:
            self._recordings.append(path)
        return path

    def _compute_debug_stats(self) -> dict:
        """Compute debug statistics from collected timing data."""
        def stats_for_deque(d: deque) -> dict:
            if not d:
                return {"avg": 0, "min": 0, "max": 0, "count": 0}
            values = list(d)
            return {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        # Calculate actual FPS
        actual_fps = 0.0
        if len(self._frame_times) >= 2:
            frame_times = list(self._frame_times)
            total_time = frame_times[-1] - frame_times[0]
            if total_time > 0:
                actual_fps = (len(frame_times) - 1) / total_time

        return {
            "actual_fps": actual_fps,
            "target_fps": self.fps,
            "fps_ratio": actual_fps / self.fps if self.fps > 0 else 0,
            "capture_ms": {k: v * 1000 for k, v in stats_for_deque(self._debug_capture_times).items()},
            "detect_ms": {k: v * 1000 for k, v in stats_for_deque(self._debug_detect_times).items()},
            "write_ms": {k: v * 1000 for k, v in stats_for_deque(self._debug_write_times).items()},
            "loop_ms": {k: v * 1000 for k, v in stats_for_deque(self._debug_loop_times).items()},
            "sleep_drift_ms": {k: v * 1000 for k, v in stats_for_deque(self._debug_sleep_drifts).items()},
            "overruns": self._debug_overruns,
            "total_frames": self._debug_total_frames,
            "is_recording": self._recorder.is_recording,
        }

    def _output_debug_stats(self) -> None:
        """Output debug stats if enabled and interval has passed."""
        if not self.debug or self.on_debug_stats is None:
            return

        now = time.time()
        if now - self._debug_last_stats_time >= self.debug_interval:
            self._debug_last_stats_time = now
            stats = self._compute_debug_stats()
            self.on_debug_stats(stats)

    def start(self) -> None:
        """Start the capture loop (blocking)."""
        self._running = True
        self._frame_times.clear()
        self._last_frame_drop_notification = 0.0
        self._debug_last_stats_time = time.time()
        self._debug_overruns = 0
        self._debug_total_frames = 0
        interval = 1.0 / self.fps

        while self._running:
            loop_start = time.time()

            # Track frame time for drop detection
            self._frame_times.append(loop_start)
            self._debug_total_frames += 1

            # --- Capture timing ---
            capture_start = time.time()
            frame = self.capture.capture_frame()
            if self.debug:
                self._debug_capture_times.append(time.time() - capture_start)

            if frame is not None:
                # --- Detection timing ---
                if self._detector is not None:
                    detect_start = time.time()
                    trigger = self._detector.check_frame(frame)
                    if self.debug:
                        self._debug_detect_times.append(time.time() - detect_start)

                    if trigger is True and not self._recorder.is_recording:
                        path = self._start_recording()
                        if self.on_recording_change:
                            self.on_recording_change(True, path)
                    elif trigger is False and self._recorder.is_recording:
                        path = self._stop_recording()
                        if self.on_recording_change:
                            self.on_recording_change(False, path)

                # --- Write timing ---
                if self._recorder.is_recording:
                    write_start = time.time()
                    self._recorder.write_frame(frame)
                    if self.debug:
                        self._debug_write_times.append(time.time() - write_start)

            # Check for frame drops (only while recording)
            if self._recorder.is_recording:
                self._check_frame_drops()

            # --- Loop timing and sleep ---
            elapsed = time.time() - loop_start
            if self.debug:
                self._debug_loop_times.append(elapsed)

            sleep_time = interval - elapsed
            if sleep_time > 0:
                sleep_start = time.time()
                time.sleep(sleep_time)
                if self.debug:
                    actual_sleep = time.time() - sleep_start
                    drift = actual_sleep - sleep_time
                    self._debug_sleep_drifts.append(drift)
            else:
                # Loop overrun - we're behind schedule
                if self.debug:
                    self._debug_overruns += 1

            # Output debug stats periodically
            self._output_debug_stats()

    def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False
        # Stop any active recording
        if self._recorder.is_recording:
            path = self._recorder.stop()
            if path:
                self._recordings.append(path)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder.is_recording

    @property
    def recordings(self) -> list[Path]:
        """List of completed recordings."""
        return self._recordings.copy()

    @property
    def is_running(self) -> bool:
        """Check if capture loop is running."""
        return self._running
