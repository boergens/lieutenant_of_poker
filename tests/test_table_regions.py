"""Tests for table region detection module."""

import numpy as np
import pytest

from lieutenant_of_poker.table_regions import (
    Region,
    PlayerPosition,
    PlayerRegions,
    TableRegionDetector,
    detect_table_regions,
)


class TestRegion:
    """Tests for Region dataclass."""

    def test_properties(self):
        """Test region property calculations."""
        region = Region(x=100, y=200, width=50, height=30)
        assert region.x2 == 150
        assert region.y2 == 230
        assert region.center == (125, 215)

    def test_extract(self):
        """Test region extraction from frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:40, 30:60] = 255  # White rectangle

        region = Region(x=30, y=20, width=30, height=20)
        extracted = region.extract(frame)

        assert extracted.shape == (20, 30, 3)
        assert np.all(extracted == 255)

    def test_scale(self):
        """Test region scaling."""
        region = Region(x=100, y=200, width=50, height=30)
        scaled = region.scale(2.0, 0.5)

        assert scaled.x == 200
        assert scaled.y == 100
        assert scaled.width == 100
        assert scaled.height == 15

    def test_contains_point(self):
        """Test point containment check."""
        region = Region(x=100, y=200, width=50, height=30)

        assert region.contains_point(125, 215)  # Center
        assert region.contains_point(100, 200)  # Top-left corner
        assert not region.contains_point(150, 230)  # Bottom-right (exclusive)
        assert not region.contains_point(99, 200)  # Just outside left


class TestTableRegionDetector:
    """Tests for TableRegionDetector class."""

    def test_initialization(self):
        """Test detector initialization with frame dimensions."""
        detector = TableRegionDetector(1728, 1117)
        assert detector.frame_width == 1728
        assert detector.frame_height == 1117
        assert detector.scale_x == 1.0
        assert detector.scale_y == 1.0

    def test_scaling(self):
        """Test that regions scale correctly for different resolutions."""
        # Double resolution
        detector = TableRegionDetector(3456, 2234)
        assert detector.scale_x == 2.0
        assert detector.scale_y == 2.0

        # Pot region should be scaled
        pot = detector.pot_region
        assert pot.x == 760 * 2
        assert pot.y == 355 * 2

    def test_all_regions_exist(self):
        """Test that all expected regions are defined."""
        detector = TableRegionDetector(1728, 1117)

        assert detector.pot_region is not None
        assert detector.community_cards_region is not None
        assert detector.hero_cards_region is not None
        assert detector.action_buttons_region is not None
        assert detector.dealer_button_search_region is not None
        assert detector.balance_region is not None

    def test_player_regions(self):
        """Test that all player positions have regions."""
        detector = TableRegionDetector(1728, 1117)

        for position in PlayerPosition:
            player_region = detector.get_player_region(position)
            assert player_region is not None
            assert player_region.position == position
            assert player_region.name_chip_box is not None

    def test_hero_has_cards(self):
        """Test that hero position has cards region."""
        detector = TableRegionDetector(1728, 1117)
        hero = detector.get_player_region(PlayerPosition.HERO)
        assert hero.cards is not None

    def test_extract_methods(self):
        """Test region extraction methods."""
        detector = TableRegionDetector(1728, 1117)
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)

        pot = detector.extract_pot(frame)
        assert pot.shape[0] > 0 and pot.shape[1] > 0

        community = detector.extract_community_cards(frame)
        assert community.shape[0] > 0 and community.shape[1] > 0

        hero = detector.extract_hero_cards(frame)
        assert hero.shape[0] > 0 and hero.shape[1] > 0

    def test_draw_regions(self):
        """Test that draw_regions returns annotated frame."""
        detector = TableRegionDetector(1728, 1117)
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)

        annotated = detector.draw_regions(frame)
        assert annotated.shape == frame.shape
        # Should have drawn something (not all zeros)
        assert not np.array_equal(annotated, frame)


class TestDetectTableRegions:
    """Tests for detect_table_regions helper function."""

    def test_creates_detector(self):
        """Test that function creates detector from frame."""
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)
        detector = detect_table_regions(frame)

        assert isinstance(detector, TableRegionDetector)
        assert detector.frame_width == 1728
        assert detector.frame_height == 1117
