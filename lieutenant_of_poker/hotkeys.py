"""
Global hotkey handling for macOS.

Provides a simple interface for registering global keyboard shortcuts.
"""

import sys
from typing import Callable, Optional


class HotkeyListener:
    """
    Listens for global keyboard shortcuts.

    Uses pynput for cross-platform hotkey detection.
    """

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        """
        Initialize the hotkey listener.

        Args:
            hotkey: Hotkey string like "cmd+shift+r" or "ctrl+alt+s".
            callback: Function to call when hotkey is pressed.
        """
        self.hotkey_str = hotkey
        self.callback = callback
        self._listener = None
        self._modifiers = set()
        self._key = None
        self._current_modifiers = set()

        self._parse_hotkey(hotkey)

    def _parse_hotkey(self, hotkey: str) -> None:
        """Parse a hotkey string into modifiers and key."""
        from pynput import keyboard

        parts = hotkey.lower().split("+")

        for part in parts:
            part = part.strip()
            if part in ("cmd", "command", "super"):
                self._modifiers.add(keyboard.Key.cmd)
            elif part in ("ctrl", "control"):
                self._modifiers.add(keyboard.Key.ctrl)
            elif part in ("shift",):
                self._modifiers.add(keyboard.Key.shift)
            elif part in ("alt", "option"):
                self._modifiers.add(keyboard.Key.alt)
            elif len(part) == 1:
                self._key = keyboard.KeyCode.from_char(part)
            else:
                raise ValueError(f"Unknown hotkey part: '{part}'")

        if self._key is None:
            raise ValueError(f"No key specified in hotkey: '{hotkey}'")

    def start(self) -> None:
        """Start listening for the hotkey."""
        from pynput import keyboard

        def on_press(k):
            if isinstance(k, keyboard.Key):
                self._current_modifiers.add(k)
            elif k == self._key and self._modifiers <= self._current_modifiers:
                self.callback()

        def on_release(k):
            if isinstance(k, keyboard.Key):
                self._current_modifiers.discard(k)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        """Stop listening."""
        if self._listener:
            self._listener.stop()
            self._listener = None


def play_sound(sound_name: str) -> None:
    """
    Play a system sound (macOS only).

    Args:
        sound_name: Name of sound file in /System/Library/Sounds/
                   e.g., "Glass", "Blow", "Ping", "Pop"
    """
    try:
        import subprocess
        sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
        subprocess.run(["afplay", sound_path], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def create_hotkey_listener(
    hotkey: str,
    callback: Callable[[], None],
) -> Optional[HotkeyListener]:
    """
    Create and start a hotkey listener.

    Args:
        hotkey: Hotkey string like "cmd+shift+r".
        callback: Function to call when pressed.

    Returns:
        HotkeyListener instance, or None if setup failed.
    """
    try:
        listener = HotkeyListener(hotkey, callback)
        listener.start()
        return listener
    except ImportError:
        print("Warning: pynput not installed, hotkey disabled", file=sys.stderr)
        print("  Install with: pip install pynput", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Warning: Invalid hotkey '{hotkey}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Could not set up hotkey: {e}", file=sys.stderr)
        return None
