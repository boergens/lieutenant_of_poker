"""
Card detection and recognition for Governor of Poker.

Detects playing cards from frame regions and identifies rank and suit
using OCR and color analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np

from .fast_ocr import ocr_card_rank


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
        return f"{self.rank.value}{suit_chars[self.suit]}"


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

    def __init__(self):
        """Initialize the card detector."""
        # Color thresholds for suit detection (in HSV)
        # Red for hearts/diamonds
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

        # Black for clubs/spades
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 80])

    def detect_card(self, card_image: np.ndarray) -> Optional[Card]:
        """
        Detect a single card from an image region.

        Args:
            card_image: BGR image of a single card.

        Returns:
            Card object if detected, None otherwise.
        """
        if card_image is None or card_image.size == 0:
            return None

        rank = self._detect_rank(card_image)
        suit = self._detect_suit(card_image)

        if rank is None or suit is None:
            return None

        return Card(rank=rank, suit=suit)

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
        for rect in card_rects:
            x, y, w, h = rect
            card_img = cards_region[y:y+h, x:x+w]
            card = self.detect_card(card_img)
            if card:
                cards.append(card)

        return cards

    def _detect_rank(self, card_image: np.ndarray) -> Optional[Rank]:
        """Detect the rank of a card using OCR."""
        # Scale up the card for better OCR
        scaled = cv2.resize(card_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Use primary threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Try OCR with fast tesserocr
        text = ocr_card_rank(thresh)

        # Clean up text
        text = text.upper().replace('O', '0').replace('I', '1')

        # Check for rank match
        for char in text:
            if char in RANK_MAP:
                return RANK_MAP[char]

        # Check for "10"
        if "10" in text or "1O" in text or "IO" in text:
            return Rank.TEN

        # Fallback: try inverted threshold
        _, thresh_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        text = ocr_card_rank(thresh_inv)
        text = text.upper().replace('O', '0').replace('I', '1')

        for char in text:
            if char in RANK_MAP:
                return RANK_MAP[char]

        if "10" in text or "1O" in text or "IO" in text:
            return Rank.TEN

        return None

    def _detect_suit(self, card_image: np.ndarray) -> Optional[Suit]:
        """Detect the suit of a card using color analysis."""
        # Convert to HSV
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

        # Create masks for red and black
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)

        # Count red and black pixels
        red_pixels = cv2.countNonZero(red_mask)
        black_pixels = cv2.countNonZero(black_mask)

        # Determine if red or black suit
        is_red = red_pixels > black_pixels * 0.1  # Red suits have significant red

        if is_red:
            # Distinguish hearts from diamonds by shape
            # Hearts are more rounded, diamonds are more angular
            return self._classify_red_suit(card_image, red_mask)
        else:
            # Distinguish clubs from spades by shape
            return self._classify_black_suit(card_image, black_mask)

    def _classify_red_suit(self, card_image: np.ndarray, red_mask: np.ndarray) -> Suit:
        """Classify between hearts and diamonds."""
        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return Suit.HEARTS  # Default

        # Get the largest contour (likely the suit symbol)
        largest = max(contours, key=cv2.contourArea)

        # Analyze shape - diamonds are more angular
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # Diamonds typically approximate to 4 points, hearts to more
        if len(approx) <= 6:
            return Suit.DIAMONDS
        else:
            return Suit.HEARTS

    def _classify_black_suit(self, card_image: np.ndarray, black_mask: np.ndarray) -> Suit:
        """Classify between clubs and spades."""
        # Find contours
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return Suit.SPADES  # Default

        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)

        # Analyze shape
        # Spades have a pointed top, clubs are more rounded
        x, y, w, h = cv2.boundingRect(largest)

        # Check the top portion of the bounding box
        top_region = black_mask[y:y+h//3, x:x+w]
        top_pixels = cv2.countNonZero(top_region)
        total_top = (h//3) * w

        # Spades fill more of the top (pointed), clubs less (rounded lobes)
        fill_ratio = top_pixels / total_top if total_top > 0 else 0

        if fill_ratio > 0.3:
            return Suit.SPADES
        else:
            return Suit.CLUBS

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
