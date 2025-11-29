"""Tests for card matching module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import cv2

from lieutenant_of_poker.card_matcher import CardMatcher
from lieutenant_of_poker.card_detector import Card, Rank, Suit


class TestCardMatcher:
    """Tests for CardMatcher class."""

    @pytest.fixture
    def temp_library(self, tmp_path):
        """Create a temporary card library directory."""
        return tmp_path / "card_library"

    def test_initialization(self, temp_library):
        """Test matcher initializes and creates library directory."""
        matcher = CardMatcher(library_dir=temp_library)
        assert matcher is not None
        assert temp_library.exists()

    def test_parse_filename_valid(self, temp_library):
        """Test parsing valid card filenames."""
        matcher = CardMatcher(library_dir=temp_library)

        card = matcher._parse_filename("Q_hearts.png")
        assert card is not None
        assert card.rank == Rank.QUEEN
        assert card.suit == Suit.HEARTS

        card = matcher._parse_filename("A_spades.png")
        assert card is not None
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

        card = matcher._parse_filename("10_diamonds.png")
        assert card is not None
        assert card.rank == Rank.TEN
        assert card.suit == Suit.DIAMONDS

    def test_parse_filename_invalid(self, temp_library):
        """Test parsing invalid card filenames."""
        matcher = CardMatcher(library_dir=temp_library)

        assert matcher._parse_filename("invalid.png") is None
        assert matcher._parse_filename("X_hearts.png") is None
        assert matcher._parse_filename("Q_invalid.png") is None

    def test_normalize_image(self, temp_library):
        """Test image normalization."""
        matcher = CardMatcher(library_dir=temp_library)

        # Create test image
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        img[:, :] = (128, 128, 128)

        normalized = matcher._normalize_image(img)

        # Check size is standard
        assert normalized.shape[:2] == (90, 60)
        # Check values are normalized to 0-1
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

    def test_compare_images_identical(self, temp_library):
        """Test comparing identical images."""
        matcher = CardMatcher(library_dir=temp_library)

        img = np.random.rand(90, 60, 3).astype(np.float32)
        score = matcher._compare_images(img, img)

        assert score == 0.0

    def test_compare_images_different(self, temp_library):
        """Test comparing different images."""
        matcher = CardMatcher(library_dir=temp_library)

        img1 = np.zeros((90, 60, 3), dtype=np.float32)
        img2 = np.ones((90, 60, 3), dtype=np.float32)

        score = matcher._compare_images(img1, img2)
        assert score > 0.5  # Very different

    def test_save_and_load_library(self, temp_library):
        """Test saving card to library and loading it back."""
        # Create a unique test image
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        img[20:80, 20:60] = (255, 255, 255)  # White rectangle

        matcher = CardMatcher(library_dir=temp_library)

        # Manually save a card to the library
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        matcher._save_to_library(img, card)

        # Reload the library
        matcher._load_library()

        # Check the card is in the library
        key = (Rank.ACE, Suit.SPADES)
        assert key in matcher._library

    def test_get_library_stats(self, temp_library):
        """Test library statistics."""
        matcher = CardMatcher(library_dir=temp_library)

        stats = matcher.get_library_stats()
        assert "total_images" in stats
        assert "unique_cards" in stats
        assert "cards" in stats

    def test_match_empty_image(self, temp_library):
        """Test matching with empty image."""
        matcher = CardMatcher(library_dir=temp_library)

        result = matcher.match_card(None)
        assert result is None

        result = matcher.match_card(np.array([]))
        assert result is None
