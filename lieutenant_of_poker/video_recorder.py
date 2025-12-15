"""
Video recording from frame sources.

Provides a simple VideoRecorder that can record frames from ScreenCaptureKitCapture
to a video file.
"""

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2

from .frame_extractor import FrameInfo
from ._detection import detect_hero_cards


class VideoRecorder:
    """
    Records frames to a video file.

    Buffers the last few frames so they can be discarded when recording stops
    (useful when stop is triggered by detecting something in those frames).
    """

    def __init__(self, fps: int = 10, codec: str = "mp4v", trailing_frames_to_drop: int = 3):
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

    def stop(self) -> Optional[Path]:
        """Stop recording and return the output path. Discards buffered trailing frames."""
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
                str(self._recording_path), fourcc, self.fps, (width, height)
            )
            if not self._writer.isOpened():
                print(f"ERROR: Failed to open VideoWriter for {self._recording_path}", file=sys.stderr)

        # Buffer frames - only write when buffer is full
        if len(self._frame_buffer) == self.trailing_frames_to_drop:
            self._writer.write(self._frame_buffer[0])
            self._frame_count += 1

        self._frame_buffer.append(image.copy())

    @property
    def is_recording(self) -> bool:
        return self._recording_path is not None

    @property
    def current_path(self) -> Optional[Path]:
        return self._recording_path


def run_recording_session(
    output_dir: str,
    fps: int,
    hotkey: str,
    prefix: str,
    auto: bool,
    debug: bool,
) -> None:
    """Run an interactive recording session."""
    from .screen_capture import (
        ScreenCaptureKitCapture,
        check_screen_recording_permission,
        get_permission_instructions,
        get_candidate_windows,
    )
    from .hotkeys import create_hotkey_listener, play_sound
    from .notifications import OverlayNotifier

    if not check_screen_recording_permission():
        print("Error: Screen recording permission not granted.", file=sys.stderr)
        print(get_permission_instructions(), file=sys.stderr)
        sys.exit(1)

    # Get candidate windows and let user choose
    candidates = get_candidate_windows()
    if not candidates:
        print("Error: No suitable windows found.", file=sys.stderr)
        print("Make sure you have a window open that's at least 800x600.", file=sys.stderr)
        sys.exit(1)

    print("\nAvailable windows:", file=sys.stderr)
    for i, win in enumerate(candidates, 1):
        width, height = win.bounds[2], win.bounds[3]
        print(f"  {i}. {win.owner_name}: {win.title} (video: {width}x{height})", file=sys.stderr)

    if len(candidates) == 1:
        selected = candidates[0]
        print(f"\nAuto-selecting only available window.", file=sys.stderr)
    else:
        print(f"\nSelect window [1-{len(candidates)}]: ", file=sys.stderr, end="", flush=True)
        try:
            choice = input().strip()
            idx = int(choice) - 1
            if idx < 0 or idx >= len(candidates):
                print("Invalid selection.", file=sys.stderr)
                sys.exit(1)
            selected = candidates[idx]
        except (ValueError, EOFError):
            print("Invalid selection.", file=sys.stderr)
            sys.exit(1)

    # Check window dimensions - wait for user to resize if needed
    target_width, target_height = 1342, 960
    current_width, current_height = selected.bounds[2], selected.bounds[3]

    if current_width != target_width or current_height != target_height:
        print(f"\nWindow size: {current_width}x{current_height}", file=sys.stderr)
        print(f"Expected size: {target_width}x{target_height}", file=sys.stderr)
        print(f"\nPlease resize the window manually. Waiting for correct size...", file=sys.stderr)

        stable_start = None
        stable_duration = 1.0  # seconds window must stay at correct size

        while True:
            time.sleep(0.2)
            # Re-fetch window info
            updated_windows = get_candidate_windows(min_width=100, min_height=100)
            current_win = None
            for w in updated_windows:
                if w.window_id == selected.window_id:
                    current_win = w
                    break

            if current_win is None:
                print("\rWindow closed or not found.", file=sys.stderr)
                sys.exit(1)

            current_width, current_height = current_win.bounds[2], current_win.bounds[3]

            if current_width == target_width and current_height == target_height:
                if stable_start is None:
                    stable_start = time.time()
                elapsed = time.time() - stable_start
                remaining = stable_duration - elapsed
                if remaining <= 0:
                    print(f"\r{current_width}x{current_height} - Correct! Proceeding...          ", file=sys.stderr)
                    break
                else:
                    print(f"\r{current_width}x{current_height} - Correct! Hold for {remaining:.1f}s...   ", file=sys.stderr, end="", flush=True)
            else:
                stable_start = None
                print(f"\r{current_width}x{current_height} (need {target_width}x{target_height})   ", file=sys.stderr, end="", flush=True)

        print(file=sys.stderr)  # newline after the loop

    try:
        capture = ScreenCaptureKitCapture(window_id=selected.window_id)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if capture.window is None:
        print("Error: Failed to capture selected window.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCapture target: {capture.window}", file=sys.stderr)

    # Setup
    overlay = OverlayNotifier()
    recorder = VideoRecorder(fps=fps)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recordings: list[Path] = []

    # Auto-detection state
    auto_consecutive = 3
    auto_history: deque = deque(maxlen=auto_consecutive)
    auto_cards_visible = False

    # Frame drop tracking
    frame_times: deque = deque(maxlen=30)
    last_frame_drop_notification = 0.0
    frame_drop_threshold = 0.75
    frame_drop_cooldown = 5.0

    # Debug stats
    debug_capture_times: deque = deque(maxlen=100)
    debug_detect_times: deque = deque(maxlen=100)
    debug_write_times: deque = deque(maxlen=100)
    debug_loop_times: deque = deque(maxlen=100)
    debug_sleep_drifts: deque = deque(maxlen=100)
    debug_last_stats_time = time.time()
    debug_overruns = 0
    debug_total_frames = 0
    debug_interval = 5.0

    def notify_recording_change(is_recording: bool, path):
        if is_recording:
            print(f"\n🔴 Recording started: {path}", file=sys.stderr)
            play_sound("Blow")
            overlay.send_message("Recording Started", str(Path(path).name))
        else:
            print(f"\n⬜ Recording stopped: {path}", file=sys.stderr)
            play_sound("Glass")
            overlay.send_message("Recording Stopped", str(Path(path).name))

    def start_recording() -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"{prefix}_{timestamp}.mp4"
        recorder.start(filename)
        return filename

    def stop_recording() -> Optional[Path]:
        path = recorder.stop()
        if path:
            recordings.append(path)
        return path

    def toggle_recording() -> tuple[bool, Optional[Path]]:
        if recorder.is_recording:
            return False, stop_recording()
        else:
            return True, start_recording()

    def on_hotkey():
        is_recording, path = toggle_recording()
        notify_recording_change(is_recording, recorder.current_path if is_recording else path)

    hotkey_listener = create_hotkey_listener(hotkey, on_hotkey)

    print(f"\nReady to record (Ctrl+C to exit)...", file=sys.stderr)
    print(f"  FPS: {fps}", file=sys.stderr)
    print(f"  Output: {output_path.absolute()}", file=sys.stderr)
    print(f"  Hotkey: {hotkey} (toggle recording)", file=sys.stderr)
    if auto:
        print(f"  Auto-detect: ENABLED (hero card detection)", file=sys.stderr)
    if debug:
        print(f"  Debug: ENABLED (stats every 5s)", file=sys.stderr)
    print(f"\nPress {hotkey} to start/stop recording...", file=sys.stderr)

    running = True
    interval = 1.0 / fps

    try:
        while running:
            loop_start = time.time()
            frame_times.append(loop_start)
            debug_total_frames += 1

            # Capture
            capture_start = time.time()
            frame = capture.capture_frame()
            if debug:
                debug_capture_times.append(time.time() - capture_start)

            if frame is not None:
                # Auto-detection
                if auto:
                    detect_start = time.time()
                    cards_found = detect_hero_cards(frame.image)
                    auto_history.append(cards_found)
                    if debug:
                        debug_detect_times.append(time.time() - detect_start)

                    if len(auto_history) >= auto_consecutive:
                        all_have_cards = all(auto_history)
                        all_no_cards = not any(auto_history)

                        if all_have_cards and not auto_cards_visible:
                            auto_cards_visible = True
                            path = start_recording()
                            notify_recording_change(True, path)
                        elif all_no_cards and auto_cards_visible:
                            auto_cards_visible = False
                            path = stop_recording()
                            notify_recording_change(False, path)

                # Write frame
                if recorder.is_recording:
                    write_start = time.time()
                    recorder.write_frame(frame)
                    if debug:
                        debug_write_times.append(time.time() - write_start)

            # Frame drop check
            if recorder.is_recording and len(frame_times) >= 10:
                ft = list(frame_times)
                total_time = ft[-1] - ft[0]
                if total_time > 0:
                    actual_fps = (len(ft) - 1) / total_time
                    if actual_fps / fps < frame_drop_threshold:
                        now = time.time()
                        if now - last_frame_drop_notification >= frame_drop_cooldown:
                            last_frame_drop_notification = now
                            pct = (actual_fps / fps) * 100
                            print(f"\n⚠️  Frame drop: {actual_fps:.1f}/{fps} FPS ({pct:.0f}%)", file=sys.stderr)
                            play_sound("Basso")
                            overlay.send_message("⚠️ Dropping Frames", f"{actual_fps:.1f} FPS")

            # Loop timing
            elapsed = time.time() - loop_start
            if debug:
                debug_loop_times.append(elapsed)

            sleep_time = interval - elapsed
            if sleep_time > 0:
                sleep_start = time.time()
                time.sleep(sleep_time)
                if debug:
                    debug_sleep_drifts.append(time.time() - sleep_start - sleep_time)
            elif debug:
                debug_overruns += 1

            # Debug stats output
            if debug and time.time() - debug_last_stats_time >= debug_interval:
                debug_last_stats_time = time.time()
                ft = list(frame_times)
                actual_fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) >= 2 and ft[-1] > ft[0] else 0
                rec = "REC" if recorder.is_recording else "---"

                def avg_max(d):
                    return (sum(d) / len(d) * 1000, max(d) * 1000) if d else (0, 0)

                cap_avg, cap_max = avg_max(debug_capture_times)
                det_avg, det_max = avg_max(debug_detect_times)
                wrt_avg, wrt_max = avg_max(debug_write_times)
                loop_avg, loop_max = avg_max(debug_loop_times)
                drift_avg, drift_max = avg_max(debug_sleep_drifts)

                print(f"\n[DEBUG {rec}] FPS: {actual_fps:.1f}/{fps} ({actual_fps/fps*100:.0f}%) | "
                      f"frames: {debug_total_frames} | overruns: {debug_overruns}", file=sys.stderr)
                print(f"  capture: {cap_avg:.1f}ms avg, {cap_max:.1f}ms max", file=sys.stderr)
                if debug_detect_times:
                    print(f"  detect:  {det_avg:.1f}ms avg, {det_max:.1f}ms max", file=sys.stderr)
                if debug_write_times:
                    print(f"  write:   {wrt_avg:.1f}ms avg, {wrt_max:.1f}ms max", file=sys.stderr)
                print(f"  loop:    {loop_avg:.1f}ms avg, {loop_max:.1f}ms max (target: {1000/fps:.1f}ms)", file=sys.stderr)
                print(f"  drift:   {drift_avg:.1f}ms avg, {drift_max:.1f}ms max", file=sys.stderr)

    except KeyboardInterrupt:
        pass
    finally:
        if hotkey_listener:
            hotkey_listener.stop()
        if recorder.is_recording:
            path = recorder.stop()
            if path:
                recordings.append(path)

        print(f"\nSession ended.", file=sys.stderr)
        if recordings:
            print(f"Recordings saved ({len(recordings)}):", file=sys.stderr)
            for rec in recordings:
                print(f"  - {rec}", file=sys.stderr)
        else:
            print("No recordings made.", file=sys.stderr)
