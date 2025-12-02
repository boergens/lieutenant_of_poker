"""
Card detection and recognition for Governor of Poker.

Detects playing cards from frame regions using a reference library.
Unmatched images are saved to an 'unmatched' subfolder for manual review.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Path to reference table background color image
TABLE_COLOR_IMAGE = Path(__file__).parent / "card_library" / "table_background.png"


class Suit(Enum):
    """Card suits."""
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class Rank(Enum):
    """Card ranks."""
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


@dataclass
class Card:
    """A playing card with rank and suit."""
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        suit_symbols = {
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
            Suit.SPADES: "♠",
        }
        return f"{self.rank.value}{suit_symbols[self.suit]}"

    @property
    def short_name(self) -> str:
        """Short name like 'Ah' for Ace of Hearts."""
        suit_chars = {
            Suit.HEARTS: "h",
            Suit.DIAMONDS: "d",
            Suit.CLUBS: "c",
            Suit.SPADES: "s",
        }
        # Use 'T' for ten (standard poker notation)
        rank_char = "T" if self.rank == Rank.TEN else self.rank.value
        return f"{rank_char}{suit_chars[self.suit]}"


# Mapping of OCR text to ranks
RANK_MAP = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "0": Rank.TEN,  # Sometimes OCR reads 10 as 0
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}


class CardDetector:
    """Detects and recognizes playing cards from image regions."""

    # Default table background color (purple/magenta from Governor of Poker)
    DEFAULT_TABLE_COLOR_BGR = np.array([204, 96, 184])
    # Maximum color distance to consider a slot empty
    EMPTY_SLOT_THRESHOLD = 60

    def __init__(self, use_library: bool = True):
        """
        Initialize the card detector.

        Args:
            use_library: If True, use the card library for matching.
                        If False, fall back to color-based detection only.
        """
        # Load table background color from reference image
        self.table_color = self._load_table_color()
        self.use_library = use_library

        # Color thresholds for suit detection (in HSV) - fallback only
        # Red for hearts/diamonds
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

        # Black for clubs/spades
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 80])

    def _match_card(self, card_image: np.ndarray, slot_index: int) -> Optional[Card]:
        """Match a card using the appropriate library based on slot index."""
        if not self.use_library:
            return None
        from .card_matcher import match_card
        return match_card(card_image, slot_index)

    def _load_table_color(self) -> np.ndarray:
        """Load the table background color from reference image."""
        if TABLE_COLOR_IMAGE.exists():
            img = cv2.imread(str(TABLE_COLOR_IMAGE))
            if img is not None:
                # Get average color
                return np.mean(img, axis=(0, 1))
        return self.DEFAULT_TABLE_COLOR_BGR

    def is_empty_slot(self, slot_image: np.ndarray) -> bool:
        """
        Check if a card slot is empty (shows table background).

        Args:
            slot_image: BGR image of a card slot.

        Returns:
            True if the slot appears empty (matches table color).
        """
        if slot_image is None or slot_image.size == 0:
            return True

        # Get average color of the slot
        avg_color = np.mean(slot_image, axis=(0, 1))

        # Calculate Euclidean distance to table color
        distance = np.linalg.norm(avg_color - self.table_color)

        return distance < self.EMPTY_SLOT_THRESHOLD

    def detect_card(self, card_image: np.ndarray, slot_index: int = 0) -> Optional[Card]:
        """
        Detect a single card from an image region.

        Uses the card library for matching. Unmatched images are saved
        to an 'unmatched' subfolder for manual review.

        Args:
            card_image: BGR image of a single card.
            slot_index: Which slot this card is from (for library naming).

        Returns:
            Card object if detected, None otherwise.
        """
        if card_image is None or card_image.size == 0:
            return None

        # Skip if slot appears empty (matches table background)
        if self.is_empty_slot(card_image):
            return None

        # Use the card library for matching
        if self.use_library:
            return self._match_card(card_image, slot_index)

        # Fallback to color-based detection (no rank detection without library)
        return None

    def detect_cards(self, cards_region: np.ndarray, expected_count: int = 0) -> List[Card]:
        """
        Detect multiple cards from a region containing several cards.

        Args:
            cards_region: BGR image containing multiple cards.
            expected_count: Expected number of cards (0 for auto-detect).

        Returns:
            List of detected Card objects.
        """
        if cards_region is None or cards_region.size == 0:
            return []

        # Find individual card regions
        card_rects = self._find_card_rectangles(cards_region)

        cards = []
        for i, rect in enumerate(card_rects):
            x, y, w, h = rect
            card_img = cards_region[y:y+h, x:x+w]
            card = self.detect_card(card_img, slot_index=i)
            if card:
                cards.append(card)

        return cards

    def _find_card_rectangles(self, region: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find individual card rectangles in a multi-card region."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Threshold to find card edges (cards are white/light)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort by x position (left to right)
        rects = []
        h, w = region.shape[:2]
        min_area = (h * w) * 0.01  # At least 1% of region
        min_height = h * 0.3  # Cards should be at least 30% of region height

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            if area > min_area and ch > min_height:
                rects.append((x, y, cw, ch))

        # Sort by x position
        rects.sort(key=lambda r: r[0])

        return rects


def detect_cards_in_region(region: np.ndarray) -> List[Card]:
    """
    Convenience function to detect cards in a region.

    Args:
        region: BGR image region containing cards.

    Returns:
        List of detected Card objects.
    """
    detector = CardDetector()
    return detector.detect_cards(region)
