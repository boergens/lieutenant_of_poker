"""Tests for card matching module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import cv2

from lieutenant_of_poker.card_matcher import (
    CardMatcher, RankMatcher, SuitMatcher,
    RANK_REGION, SUIT_REGION,
    COMMUNITY_LIBRARY, HERO_LEFT_LIBRARY, HERO_RIGHT_LIBRARY,
    get_card_matcher, match_card,
)
from lieutenant_of_poker.card_detector import Card, Rank, Suit


class TestRankMatcher:
    """Tests for RankMatcher class."""

    @pytest.fixture
    def temp_library(self, tmp_path):
        """Create a temporary library directory."""
        return tmp_path / "ranks"

    def test_initialization(self, temp_library):
        """Test matcher initializes and creates library directory."""
        matcher = RankMatcher(library_dir=temp_library)
        assert matcher is not None
        assert temp_library.exists()

    def test_parse_filename_valid(self, temp_library):
        """Test parsing valid rank filenames."""
        matcher = RankMatcher(library_dir=temp_library)

        assert matcher._parse_filename("Q.png") == Rank.QUEEN
        assert matcher._parse_filename("A.png") == Rank.ACE
        assert matcher._parse_filename("10.png") == Rank.TEN
        assert matcher._parse_filename("2.png") == Rank.TWO

    def test_parse_filename_invalid(self, temp_library):
        """Test parsing invalid rank filenames."""
        matcher = RankMatcher(library_dir=temp_library)

        assert matcher._parse_filename("invalid.png") is None
        assert matcher._parse_filename("X.png") is None

    def test_save_and_load_library(self, temp_library):
        """Test saving rank to library and loading it back."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = (255, 255, 255)

        matcher = RankMatcher(library_dir=temp_library)
        matcher._save_to_library(img, Rank.ACE)
        matcher._load_library()

        assert Rank.ACE in matcher._library


class TestSuitMatcher:
    """Tests for SuitMatcher class."""

    @pytest.fixture
    def temp_library(self, tmp_path):
        """Create a temporary library directory."""
        return tmp_path / "suits"

    def test_initialization(self, temp_library):
        """Test matcher initializes and creates library directory."""
        matcher = SuitMatcher(library_dir=temp_library)
        assert matcher is not None
        assert temp_library.exists()

    def test_parse_filename_valid(self, temp_library):
        """Test parsing valid suit filenames."""
        matcher = SuitMatcher(library_dir=temp_library)

        assert matcher._parse_filename("hearts.png") == Suit.HEARTS
        assert matcher._parse_filename("diamonds.png") == Suit.DIAMONDS
        assert matcher._parse_filename("clubs.png") == Suit.CLUBS
        assert matcher._parse_filename("spades.png") == Suit.SPADES

    def test_parse_filename_invalid(self, temp_library):
        """Test parsing invalid suit filenames."""
        matcher = SuitMatcher(library_dir=temp_library)

        assert matcher._parse_filename("invalid.png") is None
        assert matcher._parse_filename("X.png") is None

    def test_save_and_load_library(self, temp_library):
        """Test saving suit to library and loading it back."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = (255, 0, 0)

        matcher = SuitMatcher(library_dir=temp_library)
        matcher._save_to_library(img, Suit.HEARTS)
        matcher._load_library()

        assert Suit.HEARTS in matcher._library


class TestCardMatcher:
    """Tests for CardMatcher class."""

    def test_initialization(self):
        """Test matcher initializes with rank and suit matchers."""
        matcher = CardMatcher()
        assert matcher.rank_matcher is not None
        assert matcher.suit_matcher is not None

    def test_extract_rank_region(self):
        """Test rank region extraction."""
        matcher = CardMatcher()

        # Create a test image
        img = np.zeros((150, 110, 3), dtype=np.uint8)
        x, y, w, h = RANK_REGION
        img[y:y+h, x:x+w] = (255, 255, 255)  # White rank region

        rank_region = matcher.extract_rank_region(img)
        assert rank_region.shape == (h, w, 3)
        assert np.all(rank_region == 255)

    def test_extract_suit_region(self):
        """Test suit region extraction."""
        matcher = CardMatcher()

        # Create a test image
        img = np.zeros((150, 110, 3), dtype=np.uint8)
        x, y, w, h = SUIT_REGION
        img[y:y+h, x:x+w] = (0, 0, 255)  # Red suit region

        suit_region = matcher.extract_suit_region(img)
        assert suit_region.shape == (h, w, 3)
        assert np.all(suit_region == (0, 0, 255))

    def test_get_library_stats(self):
        """Test library statistics."""
        matcher = CardMatcher()
        stats = matcher.get_library_stats()

        assert "library_name" in stats
        assert "ranks" in stats
        assert "suits" in stats
        assert "total_possible" in stats
        assert stats["total_possible"] == 17  # 13 ranks + 4 suits
        assert stats["library_name"] == COMMUNITY_LIBRARY

    def test_library_name_parameter(self, tmp_path):
        """Test creating matchers with different library names."""
        # Create matchers with different library names using tmp directory
        community_matcher = CardMatcher(library_name=COMMUNITY_LIBRARY)
        left_matcher = CardMatcher(library_name=HERO_LEFT_LIBRARY)
        right_matcher = CardMatcher(library_name=HERO_RIGHT_LIBRARY)

        # Verify they have different library names
        assert community_matcher.library_name == COMMUNITY_LIBRARY
        assert left_matcher.library_name == HERO_LEFT_LIBRARY
        assert right_matcher.library_name == HERO_RIGHT_LIBRARY

        # Verify they use different directories
        assert community_matcher.rank_matcher.library_dir != left_matcher.rank_matcher.library_dir
        assert left_matcher.rank_matcher.library_dir != right_matcher.rank_matcher.library_dir

    def test_match_empty_image(self):
        """Test matching with empty image."""
        matcher = CardMatcher()

        result = matcher.match_card(None)
        assert result is None

        result = matcher.match_card(np.array([]))
        assert result is None
