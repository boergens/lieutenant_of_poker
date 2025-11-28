"""Tests for chip/money OCR module."""

import numpy as np
import pytest

from lieutenant_of_poker.chip_ocr import ChipOCR, extract_chip_amount


class TestChipOCR:
    """Tests for ChipOCR class."""

    def test_initialization(self):
        """Test OCR initializes without error."""
        ocr = ChipOCR()
        assert ocr is not None

    def test_extract_amount_none(self):
        """Test extraction returns None for invalid input."""
        ocr = ChipOCR()
        assert ocr.extract_amount(None) is None
        assert ocr.extract_amount(np.array([])) is None

    def test_extract_pot(self):
        """Test pot extraction method exists."""
        ocr = ChipOCR()
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = ocr.extract_pot(img)
        # May return None for blank image, but shouldn't error
        assert result is None or isinstance(result, int)

    def test_extract_player_chips(self):
        """Test player chip extraction method exists."""
        ocr = ChipOCR()
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = ocr.extract_player_chips(img)
        assert result is None or isinstance(result, int)

    def test_extract_bet(self):
        """Test bet extraction method exists."""
        ocr = ChipOCR()
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = ocr.extract_bet(img)
        assert result is None or isinstance(result, int)

    def test_parse_amount_basic(self):
        """Test basic amount parsing."""
        ocr = ChipOCR()

        # Basic numbers
        assert ocr._parse_amount("1000") == 1000
        assert ocr._parse_amount("500") == 500

        # With commas
        assert ocr._parse_amount("1,000") == 1000
        assert ocr._parse_amount("10,000") == 10000

        # With spaces (OCR sometimes adds them)
        assert ocr._parse_amount("1 000") == 1000

    def test_parse_amount_suffixes(self):
        """Test K and M suffix handling."""
        ocr = ChipOCR()

        assert ocr._parse_amount("1K") == 1000
        assert ocr._parse_amount("5K") == 5000
        assert ocr._parse_amount("1M") == 1000000
        assert ocr._parse_amount("2M") == 2000000

    def test_parse_amount_ocr_mistakes(self):
        """Test handling of common OCR mistakes."""
        ocr = ChipOCR()

        # O instead of 0
        assert ocr._parse_amount("1OO") == 100

        # I or L instead of 1
        assert ocr._parse_amount("I00") == 100
        assert ocr._parse_amount("L00") == 100

    def test_parse_amount_empty(self):
        """Test parsing empty/invalid strings."""
        ocr = ChipOCR()

        assert ocr._parse_amount("") is None
        assert ocr._parse_amount(None) is None


class TestExtractChipAmount:
    """Tests for convenience function."""

    def test_returns_optional_int(self):
        """Test function returns int or None."""
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = extract_chip_amount(img)
        assert result is None or isinstance(result, int)
