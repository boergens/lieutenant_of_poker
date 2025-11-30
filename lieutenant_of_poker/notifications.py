"""
Multi-channel notification system for poker alerts.

Provides various notification channels including terminal output, audio alerts,
visual overlays, and file logging.
"""

import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .rules_engine import RuleViolation, Severity, TrackedHand


@dataclass
class Notification:
    """A notification to be sent through channels."""

    violation: RuleViolation
    timestamp: datetime = field(default_factory=datetime.now)
    hand_context: Optional[TrackedHand] = None

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.violation}"


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """
        Send a notification through this channel.

        Args:
            notification: The notification to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this channel is available for use.

        Returns:
            True if the channel can be used, False otherwise.
        """
        pass


class TerminalNotifier(NotificationChannel):
    """Terminal output with rich formatting."""

    def __init__(self, use_rich: bool = True):
        """
        Initialize terminal notifier.

        Args:
            use_rich: Whether to use rich library for formatting.
        """
        self.use_rich = use_rich
        self._console = None

        if use_rich:
            try:
                from rich.console import Console

                self._console = Console(stderr=True)
            except ImportError:
                self.use_rich = False

    def is_available(self) -> bool:
        """Terminal output is always available."""
        return True

    def send(self, notification: Notification) -> bool:
        """Send notification to terminal."""
        if self.use_rich and self._console:
            return self._send_rich(notification)
        return self._send_plain(notification)

    def _send_rich(self, notification: Notification) -> bool:
        """Send with rich formatting."""
        from rich.panel import Panel
        from rich.text import Text

        severity = notification.violation.severity
        color_map = {
            Severity.INFO: "blue",
            Severity.WARNING: "yellow",
            Severity.ERROR: "red",
            Severity.CRITICAL: "bold red",
        }
        color = color_map.get(severity, "white")

        # Build the message
        text = Text()
        text.append(f"[{severity.name}] ", style=color)
        text.append(notification.violation.rule_name, style="bold")
        text.append(f"\n{notification.violation.message}")

        if notification.violation.suggestion:
            text.append(f"\n\nSuggestion: ", style="italic")
            text.append(notification.violation.suggestion, style="green italic")

        title = notification.timestamp.strftime("%H:%M:%S")
        panel = Panel(text, title=title, border_style=color)
        self._console.print(panel)
        return True

    def _send_plain(self, notification: Notification) -> bool:
        """Send as plain text."""
        print(str(notification), file=sys.stderr)
        return True


class AudioNotifier(NotificationChannel):
    """Audio alerts using system sounds."""

    # Default sound mapping by severity
    DEFAULT_SOUNDS = {
        Severity.INFO: "Pop",
        Severity.WARNING: "Ping",
        Severity.ERROR: "Basso",
        Severity.CRITICAL: "Sosumi",
    }

    def __init__(
        self,
        sound_mapping: Optional[dict[Severity, str]] = None,
        use_tts: bool = False,
    ):
        """
        Initialize audio notifier.

        Args:
            sound_mapping: Map of severity to sound name (macOS system sounds).
            use_tts: Whether to use text-to-speech for messages.
        """
        self.sound_mapping = sound_mapping or self.DEFAULT_SOUNDS
        self.use_tts = use_tts

    def is_available(self) -> bool:
        """Check if audio is available (macOS afplay command)."""
        try:
            result = subprocess.run(
                ["which", "afplay"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def send(self, notification: Notification) -> bool:
        """Play alert sound for the notification."""
        severity = notification.violation.severity
        sound_name = self.sound_mapping.get(severity, "Pop")

        # Try to play system sound
        sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
        try:
            # Run in background thread to not block
            thread = threading.Thread(
                target=self._play_sound,
                args=(sound_path,),
                daemon=True,
            )
            thread.start()

            # Optionally speak the message
            if self.use_tts:
                self._speak(notification.violation.message)

            return True
        except Exception:
            return False

    def _play_sound(self, sound_path: str) -> None:
        """Play a sound file."""
        try:
            subprocess.run(
                ["afplay", sound_path],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    def _speak(self, text: str) -> None:
        """Speak text using macOS say command."""
        try:
            subprocess.run(
                ["say", "-v", "Samantha", text],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass


class OverlayNotifier(NotificationChannel):
    """Visual overlay notification (macOS notification center)."""

    def __init__(self, app_name: str = "Lieutenant of Poker"):
        """
        Initialize overlay notifier.

        Args:
            app_name: Name to show in notifications.
        """
        self.app_name = app_name

    def is_available(self) -> bool:
        """Check if osascript is available for notifications."""
        try:
            result = subprocess.run(
                ["which", "osascript"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def send(self, notification: Notification) -> bool:
        """Send notification to macOS notification center."""
        title = f"{notification.violation.severity.name}: {notification.violation.rule_name}"
        message = notification.violation.message

        if notification.violation.suggestion:
            message += f"\n{notification.violation.suggestion}"

        # Use osascript to display notification
        script = f'''
        display notification "{message}" with title "{title}" subtitle "{self.app_name}"
        '''

        try:
            thread = threading.Thread(
                target=self._run_osascript,
                args=(script,),
                daemon=True,
            )
            thread.start()
            return True
        except Exception:
            return False

    def _run_osascript(self, script: str) -> None:
        """Run an AppleScript command."""
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass


class LogNotifier(NotificationChannel):
    """File logging for session review."""

    def __init__(self, log_path: Path | str):
        """
        Initialize log notifier.

        Args:
            log_path: Path to the log file.
        """
        self.log_path = Path(log_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if we can write to the log file."""
        try:
            self._ensure_directory()
            # Try to open file in append mode
            with open(self.log_path, "a") as f:
                pass
            return True
        except Exception:
            return False

    def send(self, notification: Notification) -> bool:
        """Write notification to log file."""
        try:
            timestamp = notification.timestamp.isoformat()
            violation = notification.violation

            # Format log entry
            entry_parts = [
                f"[{timestamp}]",
                f"[{violation.severity.name}]",
                f"[{violation.rule_name}]",
                violation.message,
            ]

            if violation.suggestion:
                entry_parts.append(f"| Suggestion: {violation.suggestion}")

            if violation.details:
                details_str = ", ".join(
                    f"{k}={v}" for k, v in violation.details.items()
                )
                entry_parts.append(f"| Details: {details_str}")

            # Add hand context if available
            if notification.hand_context:
                hand = notification.hand_context
                if hand.hero_cards:
                    cards = " ".join(str(c) for c in hand.hero_cards)
                    entry_parts.append(f"| Hero: {cards}")

            entry = " ".join(entry_parts) + "\n"

            with open(self.log_path, "a") as f:
                f.write(entry)

            return True
        except Exception:
            return False


class NotificationManager:
    """Manages all notification channels."""

    def __init__(self):
        """Initialize the notification manager."""
        self._channels: list[NotificationChannel] = list()
        self._severity_filter: Severity = Severity.WARNING
        self._notification_history: list[Notification] = list()
        self._max_history: int = 100

    def add_channel(self, channel: NotificationChannel) -> bool:
        """
        Add a notification channel.

        Args:
            channel: The channel to add.

        Returns:
            True if channel was added (is available), False otherwise.
        """
        if channel.is_available():
            self._channels.append(channel)
            return True
        return False

    def remove_channel(self, channel_type: type) -> bool:
        """
        Remove channels of a specific type.

        Args:
            channel_type: Type of channel to remove.

        Returns:
            True if any channels were removed.
        """
        original_count = len(self._channels)
        self._channels = [c for c in self._channels if not isinstance(c, channel_type)]
        return len(self._channels) < original_count

    def set_minimum_severity(self, severity: Severity) -> None:
        """
        Set minimum severity for notifications.

        Args:
            severity: Minimum severity to notify about.
        """
        self._severity_filter = severity

    def notify(self, notification: Notification) -> int:
        """
        Send notification to all channels.

        Args:
            notification: The notification to send.

        Returns:
            Number of channels that successfully sent the notification.
        """
        # Filter by severity
        if notification.violation.severity.value < self._severity_filter.value:
            return 0

        # Store in history
        self._notification_history.append(notification)
        if len(self._notification_history) > self._max_history:
            self._notification_history.pop(0)

        # Send to all channels
        success_count = 0
        for channel in self._channels:
            try:
                if channel.send(notification):
                    success_count += 1
            except Exception:
                pass

        return success_count

    def notify_violation(
        self,
        violation: RuleViolation,
        hand_context: Optional[TrackedHand] = None,
    ) -> int:
        """
        Convenience method to create and send notification from violation.

        Args:
            violation: The rule violation.
            hand_context: Optional hand context.

        Returns:
            Number of channels that successfully sent.
        """
        notification = Notification(
            violation=violation,
            hand_context=hand_context,
        )
        return self.notify(notification)

    def get_history(self) -> list[Notification]:
        """Get notification history."""
        return list(self._notification_history)

    def clear_history(self) -> None:
        """Clear notification history."""
        self._notification_history.clear()

    @property
    def channel_count(self) -> int:
        """Get number of active channels."""
        return len(self._channels)
