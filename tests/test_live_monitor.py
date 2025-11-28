"""Tests for live monitor module."""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lieutenant_of_poker.live_monitor import (
    LiveMonitor,
    MonitorStats,
    create_live_monitor,
)
from lieutenant_of_poker.screen_capture import ScreenCapture
from lieutenant_of_poker.live_state_tracker import (
    LiveStateTracker,
    StateUpdate,
    GameEvent,
    TrackedHand,
)
from lieutenant_of_poker.rules_engine import (
    RulesEngine,
    Rule,
    RuleContext,
    RuleViolation,
    Severity,
)
from lieutenant_of_poker.notifications import NotificationManager
from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.frame_extractor import FrameInfo


class TestMonitorStats:
    """Tests for MonitorStats dataclass."""

    def test_initial_values(self):
        """Test initial stat values."""
        stats = MonitorStats()
        assert stats.frames_processed == 0
        assert stats.hands_tracked == 0
        assert stats.violations_detected == 0
        assert stats.start_time is None

    def test_duration_seconds(self):
        """Test duration calculation."""
        stats = MonitorStats(
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 30),
        )
        assert stats.duration_seconds == 30.0

    def test_duration_seconds_no_start(self):
        """Test duration with no start time."""
        stats = MonitorStats()
        assert stats.duration_seconds == 0.0

    def test_avg_frame_time_ms(self):
        """Test average frame time calculation."""
        stats = MonitorStats(
            frames_processed=100,
            total_frame_time_ms=500.0,
        )
        assert stats.avg_frame_time_ms == 5.0

    def test_avg_frame_time_ms_no_frames(self):
        """Test average frame time with no frames."""
        stats = MonitorStats()
        assert stats.avg_frame_time_ms == 0.0

    def test_actual_fps(self):
        """Test actual FPS calculation."""
        stats = MonitorStats(
            frames_processed=150,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 10),
        )
        assert stats.actual_fps == 15.0


class TestLiveMonitor:
    """Tests for LiveMonitor class."""

    @pytest.fixture
    def mock_capture(self):
        """Create mock screen capture."""
        capture = MagicMock(spec=ScreenCapture)
        # Return None by default (no frames)
        capture.capture_frame.return_value = None
        return capture

    @pytest.fixture
    def mock_tracker(self):
        """Create mock state tracker."""
        tracker = MagicMock(spec=LiveStateTracker)
        tracker.get_hand_count.return_value = 1
        tracker.get_current_hand.return_value = TrackedHand(hand_id="test")
        tracker.process_frame.return_value = StateUpdate(
            state=GameState(street=Street.PREFLOP),
            events=[],
        )
        return tracker

    @pytest.fixture
    def mock_rules(self):
        """Create mock rules engine."""
        rules = MagicMock(spec=RulesEngine)
        rules.evaluate_all.return_value = []
        return rules

    @pytest.fixture
    def mock_notifications(self):
        """Create mock notification manager."""
        return MagicMock(spec=NotificationManager)

    @pytest.fixture
    def monitor(self, mock_capture, mock_tracker, mock_rules, mock_notifications):
        """Create monitor with all mocks."""
        return LiveMonitor(
            screen_capture=mock_capture,
            state_tracker=mock_tracker,
            rules_engine=mock_rules,
            notification_manager=mock_notifications,
            fps=10,
        )

    def test_init(self, monitor, mock_capture, mock_tracker, mock_rules, mock_notifications):
        """Test initialization."""
        assert monitor.capture == mock_capture
        assert monitor.tracker == mock_tracker
        assert monitor.rules == mock_rules
        assert monitor.notifications == mock_notifications
        assert monitor.fps == 10
        assert monitor.is_running is False

    def test_stop(self, monitor):
        """Test stopping the monitor."""
        monitor._running = True
        monitor._stats.start_time = datetime.now()

        monitor.stop()

        assert monitor.is_running is False
        assert monitor._stats.end_time is not None

    def test_get_stats(self, monitor):
        """Test getting stats."""
        stats = monitor.get_stats()
        assert isinstance(stats, MonitorStats)

    def test_process_frame_tracks_state(self, monitor, mock_tracker):
        """Test that process_frame calls tracker."""
        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        mock_tracker.process_frame.assert_called_once_with(frame)

    def test_process_frame_updates_hand_count(self, monitor, mock_tracker):
        """Test that process_frame updates hand count."""
        mock_tracker.get_hand_count.return_value = 5

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        assert monitor._stats.hands_tracked == 5

    def test_process_frame_evaluates_rules_on_hero_turn(
        self, monitor, mock_tracker, mock_rules
    ):
        """Test that rules are evaluated when it's hero's turn."""
        mock_tracker.process_frame.return_value = StateUpdate(
            state=GameState(street=Street.PREFLOP),
            events=[GameEvent.HERO_TURN],
        )

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        mock_rules.evaluate_all.assert_called_once()

    def test_process_frame_sends_notifications(
        self, monitor, mock_tracker, mock_rules, mock_notifications
    ):
        """Test that violations trigger notifications."""
        mock_tracker.process_frame.return_value = StateUpdate(
            state=GameState(street=Street.PREFLOP),
            events=[GameEvent.HERO_TURN],
        )

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
        )
        mock_rules.evaluate_all.return_value = [violation]

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        mock_notifications.notify.assert_called_once()
        assert monitor._stats.violations_detected == 1

    def test_set_state_callback(self, monitor, mock_tracker):
        """Test state update callback."""
        callback = MagicMock()
        monitor.set_state_callback(callback)

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        callback.assert_called_once()

    def test_set_violation_callback(
        self, monitor, mock_tracker, mock_rules
    ):
        """Test violation callback."""
        callback = MagicMock()
        monitor.set_violation_callback(callback)

        mock_tracker.process_frame.return_value = StateUpdate(
            state=GameState(street=Street.PREFLOP),
            events=[GameEvent.HERO_TURN],
        )

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        mock_rules.evaluate_all.return_value = [violation]

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        monitor._process_frame(frame)

        callback.assert_called_once_with(violation)

    def test_callback_errors_dont_crash(self, monitor, mock_tracker):
        """Test that callback errors are handled gracefully."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        monitor.set_state_callback(callback)

        frame = FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        # Should not raise
        monitor._process_frame(frame)


class TestCreateLiveMonitor:
    """Tests for create_live_monitor factory function."""

    @patch("lieutenant_of_poker.live_monitor.MacOSScreenCapture")
    def test_create_with_defaults(self, mock_capture_class):
        """Test creating monitor with default settings."""
        mock_capture = MagicMock()
        mock_capture_class.return_value = mock_capture

        monitor = create_live_monitor()

        assert monitor is not None
        assert monitor.fps == 10
        mock_capture_class.assert_called_once_with(
            window_title="Governor of Poker"
        )

    @patch("lieutenant_of_poker.live_monitor.MacOSScreenCapture")
    def test_create_with_custom_settings(self, mock_capture_class):
        """Test creating monitor with custom settings."""
        mock_capture = MagicMock()
        mock_capture_class.return_value = mock_capture

        monitor = create_live_monitor(
            window_title="Custom Game",
            fps=5,
        )

        assert monitor.fps == 5
        mock_capture_class.assert_called_once_with(
            window_title="Custom Game"
        )

    @patch("lieutenant_of_poker.live_monitor.MacOSScreenCapture")
    def test_create_registers_basic_rules(self, mock_capture_class):
        """Test that basic rules are registered."""
        mock_capture = MagicMock()
        mock_capture_class.return_value = mock_capture

        monitor = create_live_monitor()

        # Check that rules were registered
        rules = monitor.rules.list_rules()
        rule_names = [name for name, _, _ in rules]

        assert "fold_when_can_check" in rule_names
        assert "open_limp" in rule_names
