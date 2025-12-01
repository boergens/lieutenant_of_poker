"""
Live monitor orchestrator for real-time game monitoring.

Coordinates screen capture, state tracking, rules evaluation,
and notifications into a cohesive monitoring system.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Union

from .screen_capture import ScreenCapture, MacOSScreenCapture
from .video_recorder import VideoRecorder
from .live_state_tracker import (
    LiveStateTracker,
    GameEvent,
    StateUpdate,
)
from .rules_engine import RulesEngine, RuleContext, RuleViolation, Severity
from .notifications import NotificationManager, Notification
from .action_detector import PlayerAction


@dataclass
class MonitorStats:
    """Statistics for a monitoring session."""

    frames_processed: int = 0
    hands_tracked: int = 0
    violations_detected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_frame_time_ms: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Get monitoring duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def avg_frame_time_ms(self) -> float:
        """Get average frame processing time."""
        if self.frames_processed == 0:
            return 0.0
        return self.total_frame_time_ms / self.frames_processed

    @property
    def actual_fps(self) -> float:
        """Get actual frames per second achieved."""
        duration = self.duration_seconds
        if duration == 0:
            return 0.0
        return self.frames_processed / duration


class LiveMonitor:
    """
    Main orchestrator for live game monitoring.

    Coordinates:
    - Screen capture (frame acquisition)
    - State tracking (game state extraction and tracking)
    - Rules engine (mistake detection)
    - Notifications (alert delivery)
    """

    def __init__(
        self,
        screen_capture: ScreenCapture,
        state_tracker: LiveStateTracker,
        rules_engine: RulesEngine,
        notification_manager: NotificationManager,
        fps: int = 10,
    ):
        """
        Initialize the live monitor.

        Args:
            screen_capture: Screen capture implementation.
            state_tracker: State tracker for game state tracking.
            rules_engine: Rules engine for mistake detection.
            notification_manager: Notification manager for alerts.
            fps: Target frames per second to process.
        """
        self.capture = screen_capture
        self.tracker = state_tracker
        self.rules = rules_engine
        self.notifications = notification_manager
        self.fps = fps

        self._running = False
        self._stats = MonitorStats()
        self._on_state_update: Optional[Callable[[StateUpdate], None]] = None
        self._on_violation: Optional[Callable[[RuleViolation], None]] = None

        # Video recording (uses shared VideoRecorder)
        self._recorder = VideoRecorder(fps=fps)

    def set_state_callback(
        self, callback: Callable[[StateUpdate], None]
    ) -> None:
        """Set callback for state updates."""
        self._on_state_update = callback

    def set_violation_callback(
        self, callback: Callable[[RuleViolation], None]
    ) -> None:
        """Set callback for rule violations."""
        self._on_violation = callback

    def start_recording(
        self,
        output_path: Union[str, Path],
        codec: str = "mp4v",
    ) -> None:
        """
        Start recording captured frames to a video file.

        Args:
            output_path: Path for the output video file.
            codec: FourCC codec code (default: 'mp4v' for .mp4 files).
        """
        self._recorder.codec = codec
        self._recorder.start(output_path)

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and finalize the video file.

        Returns:
            Path to the recorded video file, or None if not recording.
        """
        return self._recorder.stop()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder.is_recording

    def start(self) -> None:
        """
        Start the monitoring loop.

        This is a blocking call that runs until stop() is called
        or an exception occurs.
        """
        self._running = True
        self._stats = MonitorStats(start_time=datetime.now())
        self._monitor_loop()

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        self._stats.end_time = datetime.now()
        # Clean up video recording if active
        self.stop_recording()

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        interval = 1.0 / self.fps

        while self._running:
            loop_start = time.time()

            # Capture frame
            frame = self.capture.capture_frame()
            if frame is None:
                time.sleep(interval)
                continue

            # Record frame if recording
            self._recorder.write_frame(frame)

            # Process frame
            frame_start = time.time()
            self._process_frame(frame)
            frame_time = (time.time() - frame_start) * 1000

            # Update stats
            self._stats.frames_processed += 1
            self._stats.total_frame_time_ms += frame_time

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self, frame) -> None:
        """Process a single captured frame."""
        # Track state
        update = self.tracker.process_frame(frame)

        # Update hand count
        self._stats.hands_tracked = self.tracker.get_hand_count()

        # Call state callback if set
        if self._on_state_update:
            try:
                self._on_state_update(update)
            except Exception:
                pass

        # Check for events that trigger rule evaluation
        if GameEvent.HERO_TURN in update.events:
            self._evaluate_rules(update)

        # Also evaluate on new hand for informational rules
        if GameEvent.NEW_HAND in update.events:
            self._on_new_hand(update)

    def _evaluate_rules(self, update: StateUpdate) -> None:
        """Evaluate rules against current game state."""
        hand = self.tracker.get_current_hand()
        if hand is None:
            return

        # Build rule context
        # Note: We don't know the action about to be taken yet
        # Rules that need this will check available actions instead
        context = RuleContext(
            current_hand=hand,
            state=update.state,
            available_actions=self._get_available_actions(update.state),
            action_about_to_take=None,
            amount_to_call=0,  # Would need detection
        )

        # Evaluate all rules
        violations = self.rules.evaluate_all(context)

        # Process violations
        for violation in violations:
            self._stats.violations_detected += 1

            # Send notification
            notification = Notification(
                violation=violation,
                hand_context=hand,
            )
            self.notifications.notify(notification)

            # Call violation callback if set
            if self._on_violation:
                try:
                    self._on_violation(violation)
                except Exception:
                    pass

    def _get_available_actions(self, state) -> list[PlayerAction]:
        """
        Determine available actions from game state.

        This is a heuristic - in practice, we'd detect action buttons.
        """
        # Default available actions based on common scenarios
        # This could be enhanced with actual button detection
        return [
            PlayerAction.FOLD,
            PlayerAction.CHECK,
            PlayerAction.CALL,
            PlayerAction.RAISE,
            PlayerAction.BET,
        ]

    def _on_new_hand(self, update: StateUpdate) -> None:
        """Handle new hand start."""
        # Could add logging or other new-hand specific logic
        pass

    def get_stats(self) -> MonitorStats:
        """Get monitoring statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if monitor is currently running."""
        return self._running


@dataclass
class MonitorConfig:
    """Configuration for live monitoring."""

    # Capture settings
    window_title: str = "Governor of Poker"
    fullscreen: bool = False
    display_id: int = 0
    fps: int = 10

    # Rules settings
    enabled_rules: Optional[list[str]] = None  # None = all rules
    min_severity: str = "warning"  # info, warning, error, critical

    # Notification settings
    terminal_output: bool = True
    audio_alerts: bool = False
    overlay: bool = False
    log_file: Optional[Path] = None

    # Recording settings
    record_to: Optional[Path] = None


def create_capture(config: MonitorConfig) -> ScreenCapture:
    """Create appropriate screen capture based on config."""
    from .screen_capture import ScreenCaptureKitCapture

    if config.fullscreen:
        return ScreenCaptureKitCapture(
            window_title=config.window_title if config.window_title != "Governor of Poker" else None,
            capture_display=True,
            display_id=config.display_id,
        )
    else:
        return MacOSScreenCapture(window_title=config.window_title)


def create_notifications(config: MonitorConfig) -> NotificationManager:
    """Create notification manager based on config."""
    from .notifications import (
        TerminalNotifier,
        AudioNotifier,
        LogNotifier,
        OverlayNotifier,
    )

    notifications = NotificationManager()

    severity_map = {
        "info": Severity.INFO,
        "warning": Severity.WARNING,
        "error": Severity.ERROR,
        "critical": Severity.CRITICAL,
    }
    notifications.set_minimum_severity(severity_map.get(config.min_severity, Severity.WARNING))

    if config.terminal_output:
        term = TerminalNotifier()
        if term.is_available():
            notifications.add_channel(term)

    if config.audio_alerts:
        audio = AudioNotifier()
        if audio.is_available():
            notifications.add_channel(audio)

    if config.log_file:
        log = LogNotifier(config.log_file)
        if log.is_available():
            notifications.add_channel(log)

    if config.overlay:
        overlay = OverlayNotifier()
        if overlay.is_available():
            notifications.add_channel(overlay)

    return notifications


def create_rules_engine(config: MonitorConfig) -> RulesEngine:
    """Create rules engine based on config."""
    from .rules import basic_rules

    rules = RulesEngine()
    basic_rules.register_all(rules)

    if config.enabled_rules is not None:
        rules.disable_all()
        for rule_name in config.enabled_rules:
            rules.enable_rule(rule_name)

    return rules


def create_live_monitor(config: MonitorConfig) -> LiveMonitor:
    """
    Create a live monitor from configuration.

    Args:
        config: Monitor configuration.

    Returns:
        Configured LiveMonitor instance.

    Raises:
        ValueError: If game window cannot be found.
        RuntimeError: If screen capture fails to initialize.
    """
    capture = create_capture(config)
    tracker = LiveStateTracker()
    rules = create_rules_engine(config)
    notifications = create_notifications(config)

    monitor = LiveMonitor(
        screen_capture=capture,
        state_tracker=tracker,
        rules_engine=rules,
        notification_manager=notifications,
        fps=config.fps,
    )

    if config.record_to:
        monitor.start_recording(config.record_to)

    return monitor


def list_available_windows() -> list[dict]:
    """List available windows for capture."""
    capture = MacOSScreenCapture()
    return [
        {
            "window_id": w.window_id,
            "title": w.title,
            "owner": w.owner_name,
            "width": w.bounds[2],
            "height": w.bounds[3],
        }
        for w in capture.list_windows()
    ]


def list_available_rules() -> list[tuple[str, bool, str]]:
    """List available rules with (name, enabled, description)."""
    from .rules import basic_rules

    rules = RulesEngine()
    basic_rules.register_all(rules)
    return rules.list_rules()
