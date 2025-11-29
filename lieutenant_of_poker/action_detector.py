"""
Player action types for Governor of Poker.

Actions are now deduced from chip changes between frames rather than
detected from UI labels.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class PlayerAction(Enum):
    """Possible player actions in poker."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    BET = auto()
    ALL_IN = auto()
    UNKNOWN = auto()


@dataclass
class DetectedAction:
    """A player action with optional amount."""
    action: PlayerAction
    amount: Optional[int] = None  # For raise/bet/call amounts
    confidence: float = 1.0

    def __str__(self) -> str:
        if self.amount:
            return f"{self.action.name} {self.amount}"
        return self.action.name
