"""Tests for notifications module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lieutenant_of_poker.notifications import (
    AudioNotifier,
    LogNotifier,
    Notification,
    NotificationChannel,
    NotificationManager,
    OverlayNotifier,
    TerminalNotifier,
)
from lieutenant_of_poker.rules_engine import (
    RuleViolation,
    Severity,
    TrackedHand,
)


class TestNotification:
    """Tests for Notification dataclass."""

    def test_notification_creation(self):
        """Test creating a notification."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
        )
        notification = Notification(violation=violation)

        assert notification.violation == violation
        assert notification.timestamp is not None
        assert notification.hand_context is None

    def test_notification_with_hand_context(self):
        """Test notification with hand context."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
        )
        hand = TrackedHand(hand_id="123")
        notification = Notification(violation=violation, hand_context=hand)

        assert notification.hand_context == hand

    def test_notification_str(self):
        """Test string representation of notification."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.ERROR,
            message="Error message",
        )
        notification = Notification(violation=violation)
        result = str(notification)

        assert "test_rule" in result
        assert "Error message" in result


class TestTerminalNotifier:
    """Tests for TerminalNotifier class."""

    def test_is_available(self):
        """Terminal output is always available."""
        notifier = TerminalNotifier(use_rich=False)
        assert notifier.is_available() is True

    def test_send_plain(self, capsys):
        """Test sending plain text notification."""
        notifier = TerminalNotifier(use_rich=False)
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
        )
        notification = Notification(violation=violation)

        result = notifier.send(notification)

        assert result is True
        captured = capsys.readouterr()
        assert "test_rule" in captured.err
        assert "Test message" in captured.err

    def test_send_rich(self):
        """Test sending rich formatted notification."""
        mock_console = MagicMock()

        notifier = TerminalNotifier(use_rich=True)
        notifier._console = mock_console

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.ERROR,
            message="Error message",
            suggestion="Fix it",
        )
        notification = Notification(violation=violation)

        result = notifier.send(notification)

        assert result is True
        mock_console.print.assert_called_once()


class TestAudioNotifier:
    """Tests for AudioNotifier class."""

    def test_default_sounds(self):
        """Test default sound mapping."""
        notifier = AudioNotifier()
        assert Severity.INFO in notifier.sound_mapping
        assert Severity.WARNING in notifier.sound_mapping
        assert Severity.ERROR in notifier.sound_mapping
        assert Severity.CRITICAL in notifier.sound_mapping

    def test_custom_sound_mapping(self):
        """Test custom sound mapping."""
        custom = {Severity.WARNING: "CustomSound"}
        notifier = AudioNotifier(sound_mapping=custom)
        assert notifier.sound_mapping[Severity.WARNING] == "CustomSound"

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test is_available when afplay exists."""
        mock_run.return_value = MagicMock(returncode=0)
        notifier = AudioNotifier()
        assert notifier.is_available() is True

    @patch("subprocess.run")
    def test_is_available_false(self, mock_run):
        """Test is_available when afplay doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1)
        notifier = AudioNotifier()
        assert notifier.is_available() is False

    @patch("threading.Thread")
    def test_send(self, mock_thread):
        """Test sending audio notification."""
        notifier = AudioNotifier()
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        notification = Notification(violation=violation)

        result = notifier.send(notification)

        assert result is True
        mock_thread.assert_called()


class TestOverlayNotifier:
    """Tests for OverlayNotifier class."""

    def test_init(self):
        """Test initialization."""
        notifier = OverlayNotifier(app_name="Test App")
        assert notifier.app_name == "Test App"

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test is_available when osascript exists."""
        mock_run.return_value = MagicMock(returncode=0)
        notifier = OverlayNotifier()
        assert notifier.is_available() is True

    @patch("subprocess.run")
    def test_is_available_false(self, mock_run):
        """Test is_available when osascript doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1)
        notifier = OverlayNotifier()
        assert notifier.is_available() is False

    @patch("threading.Thread")
    def test_send(self, mock_thread):
        """Test sending overlay notification."""
        notifier = OverlayNotifier()
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.ERROR,
            message="Error message",
            suggestion="Fix it",
        )
        notification = Notification(violation=violation)

        result = notifier.send(notification)

        assert result is True
        mock_thread.assert_called()


class TestLogNotifier:
    """Tests for LogNotifier class."""

    def test_init_creates_directory(self):
        """Test that init creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "subdir" / "test.log"
            notifier = LogNotifier(log_path)
            assert log_path.parent.exists()

    def test_is_available(self):
        """Test is_available with valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            notifier = LogNotifier(log_path)
            assert notifier.is_available() is True

    def test_send(self):
        """Test sending log notification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            notifier = LogNotifier(log_path)

            violation = RuleViolation(
                rule_name="test_rule",
                severity=Severity.WARNING,
                message="Test message",
                suggestion="Test suggestion",
                details={"key": "value"},
            )
            notification = Notification(violation=violation)

            result = notifier.send(notification)

            assert result is True
            assert log_path.exists()

            content = log_path.read_text()
            assert "test_rule" in content
            assert "Test message" in content
            assert "Test suggestion" in content
            assert "key=value" in content

    def test_send_with_hand_context(self):
        """Test sending log with hand context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            notifier = LogNotifier(log_path)

            from lieutenant_of_poker.card_detector import Card, Rank, Suit

            violation = RuleViolation(
                rule_name="test_rule",
                severity=Severity.WARNING,
                message="Test",
            )
            hand = TrackedHand(
                hand_id="123",
                hero_cards=[
                    Card(rank=Rank.ACE, suit=Suit.SPADES),
                    Card(rank=Rank.KING, suit=Suit.SPADES),
                ],
            )
            notification = Notification(violation=violation, hand_context=hand)

            result = notifier.send(notification)

            assert result is True
            content = log_path.read_text()
            assert "Hero:" in content


class TestNotificationManager:
    """Tests for NotificationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a notification manager for testing."""
        return NotificationManager()

    @pytest.fixture
    def mock_channel(self):
        """Create a mock channel."""
        channel = MagicMock(spec=NotificationChannel)
        channel.is_available.return_value = True
        channel.send.return_value = True
        return channel

    def test_add_channel(self, manager, mock_channel):
        """Test adding a channel."""
        result = manager.add_channel(mock_channel)
        assert result is True
        assert manager.channel_count == 1

    def test_add_unavailable_channel(self, manager):
        """Test adding an unavailable channel."""
        channel = MagicMock(spec=NotificationChannel)
        channel.is_available.return_value = False

        result = manager.add_channel(channel)
        assert result is False
        assert manager.channel_count == 0

    def test_remove_channel(self, manager):
        """Test removing channels by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_notifier = LogNotifier(Path(tmpdir) / "test.log")
            manager.add_channel(log_notifier)

            result = manager.remove_channel(LogNotifier)
            assert result is True
            assert manager.channel_count == 0

    def test_remove_nonexistent_channel(self, manager):
        """Test removing a channel type that doesn't exist."""
        result = manager.remove_channel(LogNotifier)
        assert result is False

    def test_set_minimum_severity(self, manager):
        """Test setting minimum severity."""
        manager.set_minimum_severity(Severity.ERROR)
        assert manager._severity_filter == Severity.ERROR

    def test_notify_filtered_by_severity(self, manager, mock_channel):
        """Test that low severity notifications are filtered."""
        manager.add_channel(mock_channel)
        manager.set_minimum_severity(Severity.ERROR)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,  # Below threshold
            message="Test",
        )
        notification = Notification(violation=violation)

        count = manager.notify(notification)

        assert count == 0
        mock_channel.send.assert_not_called()

    def test_notify_above_severity(self, manager, mock_channel):
        """Test that high severity notifications are sent."""
        manager.add_channel(mock_channel)
        manager.set_minimum_severity(Severity.WARNING)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.ERROR,  # Above threshold
            message="Test",
        )
        notification = Notification(violation=violation)

        count = manager.notify(notification)

        assert count == 1
        mock_channel.send.assert_called_once()

    def test_notify_multiple_channels(self, manager):
        """Test notifying multiple channels."""
        channel1 = MagicMock(spec=NotificationChannel)
        channel1.is_available.return_value = True
        channel1.send.return_value = True

        channel2 = MagicMock(spec=NotificationChannel)
        channel2.is_available.return_value = True
        channel2.send.return_value = True

        manager.add_channel(channel1)
        manager.add_channel(channel2)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        notification = Notification(violation=violation)

        count = manager.notify(notification)

        assert count == 2
        channel1.send.assert_called_once()
        channel2.send.assert_called_once()

    def test_notify_handles_channel_error(self, manager):
        """Test that channel errors don't stop other channels."""
        channel1 = MagicMock(spec=NotificationChannel)
        channel1.is_available.return_value = True
        channel1.send.side_effect = Exception("Error")

        channel2 = MagicMock(spec=NotificationChannel)
        channel2.is_available.return_value = True
        channel2.send.return_value = True

        manager.add_channel(channel1)
        manager.add_channel(channel2)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        notification = Notification(violation=violation)

        count = manager.notify(notification)

        assert count == 1  # Only channel2 succeeded

    def test_notify_violation(self, manager, mock_channel):
        """Test convenience method notify_violation."""
        manager.add_channel(mock_channel)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )

        count = manager.notify_violation(violation)

        assert count == 1

    def test_history(self, manager, mock_channel):
        """Test notification history."""
        manager.add_channel(mock_channel)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        notification = Notification(violation=violation)

        manager.notify(notification)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0].violation.rule_name == "test_rule"

    def test_clear_history(self, manager, mock_channel):
        """Test clearing notification history."""
        manager.add_channel(mock_channel)

        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test",
        )
        manager.notify_violation(violation)

        manager.clear_history()

        assert len(manager.get_history()) == 0

    def test_history_max_size(self, manager, mock_channel):
        """Test that history is capped at max size."""
        manager.add_channel(mock_channel)
        manager._max_history = 5

        for i in range(10):
            violation = RuleViolation(
                rule_name=f"rule_{i}",
                severity=Severity.WARNING,
                message=f"Message {i}",
            )
            manager.notify_violation(violation)

        history = manager.get_history()
        assert len(history) == 5
        # Should have the most recent 5
        assert history[0].violation.rule_name == "rule_5"
        assert history[-1].violation.rule_name == "rule_9"
