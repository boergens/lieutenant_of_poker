"""Tests for card detection module."""

import numpy as np
import pytest
import cv2

from lieutenant_of_poker.card_detector import (
    Card,
    Suit,
    Rank,
    CardDetector,
    detect_cards_in_region,
)


class TestCard:
    """Tests for Card dataclass."""

    def test_str_representation(self):
        """Test string representation of cards."""
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert str(card) == "A♠"

        card = Card(rank=Rank.KING, suit=Suit.HEARTS)
        assert str(card) == "K♥"

        card = Card(rank=Rank.TEN, suit=Suit.DIAMONDS)
        assert str(card) == "10♦"

    def test_short_name(self):
        """Test short name format."""
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert card.short_name == "Ah"

        card = Card(rank=Rank.QUEEN, suit=Suit.CLUBS)
        assert card.short_name == "Qc"


class TestSuit:
    """Tests for Suit enum."""

    def test_all_suits(self):
        """Test all suits are defined."""
        assert Suit.HEARTS.value == "hearts"
        assert Suit.DIAMONDS.value == "diamonds"
        assert Suit.CLUBS.value == "clubs"
        assert Suit.SPADES.value == "spades"


class TestRank:
    """Tests for Rank enum."""

    def test_all_ranks(self):
        """Test all ranks are defined."""
        assert Rank.TWO.value == "2"
        assert Rank.TEN.value == "10"
        assert Rank.JACK.value == "J"
        assert Rank.QUEEN.value == "Q"
        assert Rank.KING.value == "K"
        assert Rank.ACE.value == "A"


class TestCardDetector:
    """Tests for CardDetector class."""

    def test_initialization(self):
        """Test detector initializes without error."""
        detector = CardDetector()
        assert detector is not None

    def test_detect_card_empty_image(self):
        """Test detection on empty image returns None."""
        detector = CardDetector()
        result = detector.detect_card(None)
        assert result is None

        result = detector.detect_card(np.array([]))
        assert result is None

    def test_detect_cards_empty(self):
        """Test detection on empty region returns empty list."""
        detector = CardDetector()
        result = detector.detect_cards(None)
        assert result == []

        result = detector.detect_cards(np.array([]))
        assert result == []

    def test_find_card_rectangles(self):
        """Test card rectangle finding on synthetic image."""
        detector = CardDetector()

        # Create a synthetic image with two white rectangles (cards)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[10:90, 20:80] = 255  # First "card"
        img[10:90, 120:180] = 255  # Second "card"

        rects = detector._find_card_rectangles(img)
        assert len(rects) == 2
        # Should be sorted by x position
        assert rects[0][0] < rects[1][0]

    def test_detect_suit_red(self):
        """Test red suit detection."""
        detector = CardDetector()

        # Create red image (hearts/diamonds color)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 200)  # Red in BGR

        suit = detector._detect_suit(img)
        assert suit in [Suit.HEARTS, Suit.DIAMONDS]

    def test_detect_suit_black(self):
        """Test black suit detection."""
        detector = CardDetector()

        # Create dark image (clubs/spades color)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = (30, 30, 30)  # Dark gray/black

        suit = detector._detect_suit(img)
        assert suit in [Suit.CLUBS, Suit.SPADES]


class TestDetectCardsInRegion:
    """Tests for convenience function."""

    def test_returns_list(self):
        """Test function returns a list."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detect_cards_in_region(img)
        assert isinstance(result, list)
