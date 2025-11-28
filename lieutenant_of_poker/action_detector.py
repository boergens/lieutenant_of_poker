"""
Player action detection for Governor of Poker.

Detects player actions (fold, check, call, raise, all-in) from game UI.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

import cv2
import numpy as np

from .fast_ocr import ocr_general


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
    """A detected player action."""
    action: PlayerAction
    amount: Optional[int] = None  # For raise/bet/call amounts
    confidence: float = 1.0

    def __str__(self) -> str:
        if self.amount:
            return f"{self.action.name} {self.amount}"
        return self.action.name


# Keywords for action detection
ACTION_KEYWORDS = {
    PlayerAction.FOLD: ["FOLD", "FOLDED", "FOLDS"],
    PlayerAction.CHECK: ["CHECK", "CHECKED", "CHECKS"],
    PlayerAction.CALL: ["CALL", "CALLED", "CALLS"],
    PlayerAction.RAISE: ["RAISE", "RAISED", "RAISES", "RAISE TO"],
    PlayerAction.BET: ["BET", "BETS"],
    PlayerAction.ALL_IN: ["ALL-IN", "ALL IN", "ALLIN", "ALL_IN"],
}


class ActionDetector:
    """Detects player actions from game UI regions."""

    def __init__(self):
        """Initialize the action detector."""
        pass

    def detect_action_label(self, region: np.ndarray) -> Optional[DetectedAction]:
        """
        Detect an action from a player's action label region.

        These are the labels that appear near players showing their last action
        (e.g., "CHECK", "FOLD", "CALL 100").

        Args:
            region: BGR image of an action label region.

        Returns:
            DetectedAction if found, None otherwise.
        """
        if region is None or region.size == 0:
            return None

        # Preprocess for OCR
        text = self._extract_text(region)
        if not text:
            return None

        return self._parse_action(text)

    def detect_action_buttons(self, region: np.ndarray) -> List[PlayerAction]:
        """
        Detect available action buttons from the action button region.

        Args:
            region: BGR image of the action buttons area.

        Returns:
            List of available actions shown in buttons.
        """
        if region is None or region.size == 0:
            return []

        # Extract text from the button region
        text = self._extract_text(region)
        if not text:
            return []

        actions = []
        text_upper = text.upper()

        for action, keywords in ACTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_upper:
                    if action not in actions:
                        actions.append(action)
                    break

        return actions

    def detect_highlighted_button(self, region: np.ndarray) -> Optional[PlayerAction]:
        """
        Detect which action button is highlighted/selected.

        Args:
            region: BGR image of the action buttons area.

        Returns:
            The highlighted action, or None if none detected.
        """
        # This would require more sophisticated image analysis
        # to detect button highlights/glows
        # For now, return None (not implemented)
        return None

    def _extract_text(self, region: np.ndarray) -> str:
        """Extract text from a region using OCR."""
        # Scale up for better OCR
        scaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Use Otsu threshold (best general-purpose)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = ocr_general(thresh)

        # If no result, try inverted
        if not text:
            inverted = cv2.bitwise_not(thresh)
            text = ocr_general(inverted)

        return text

    def _parse_action(self, text: str) -> Optional[DetectedAction]:
        """Parse an action from OCR text."""
        if not text:
            return None

        text_upper = text.upper().strip()

        # Check for each action type
        for action, keywords in ACTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_upper:
                    # Try to extract amount for actions that have one
                    amount = self._extract_amount(text_upper)
                    return DetectedAction(action=action, amount=amount)

        return None

    def _extract_amount(self, text: str) -> Optional[int]:
        """Extract a numeric amount from action text."""
        import re

        # Look for numbers in the text
        # Handle formats like "CALL 100", "RAISE TO 500", "BET 1,000"
        text = text.replace(',', '').replace('.', '')

        match = re.search(r'(\d+)', text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        return None


def detect_player_action(region: np.ndarray) -> Optional[DetectedAction]:
    """
    Convenience function to detect a player action from a region.

    Args:
        region: BGR image region containing action text.

    Returns:
        DetectedAction if found, None otherwise.
    """
    detector = ActionDetector()
    return detector.detect_action_label(region)
