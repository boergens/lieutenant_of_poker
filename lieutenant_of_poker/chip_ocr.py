"""
Chip/money OCR module for Governor of Poker.

Extracts chip counts, pot amounts, and bet values from game UI regions.
"""

import re
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from .fast_ocr import ocr_digits


def _image_fingerprint(image: np.ndarray) -> bytes:
    """Create a fingerprint of an image for cache lookup."""
    # Resize to small fixed size for comparison
    small = cv2.resize(image, (16, 8), interpolation=cv2.INTER_AREA)
    if len(small.shape) == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Quantize to reduce noise sensitivity
    quantized = (small // 32).astype(np.uint8)
    return quantized.tobytes()


class OCRCache:
    """Per-slot cache for OCR results."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        # deque of (fingerprint, result) tuples
        self._cache: deque[Tuple[bytes, Optional[int]]] = deque(maxlen=max_size)

    def get(self, image: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Check cache for image. Returns (found, result)."""
        fp = _image_fingerprint(image)
        for cached_fp, result in self._cache:
            if cached_fp == fp:
                return True, result
        return False, None

    def put(self, image: np.ndarray, result: Optional[int]) -> None:
        """Store result in cache."""
        fp = _image_fingerprint(image)
        self._cache.append((fp, result))


class ChipOCR:
    """Extract chip/money values from game UI regions using OCR."""

    def __init__(self):
        """Initialize the chip OCR extractor."""
        pass

    def extract_amount(self, region: np.ndarray) -> Optional[int]:
        """
        Extract a chip/money amount from an image region.

        Args:
            region: BGR image region containing a chip amount.

        Returns:
            Integer amount if successfully extracted, None otherwise.
        """
        if region is None or region.size == 0:
            return None

        # Pass directly to Tesseract - it handles binarization internally
        text = ocr_digits(region)
        return self._parse_amount(text)

    def extract_pot(self, pot_region: np.ndarray) -> Optional[int]:
        """
        Extract the pot amount from the pot display region.

        Args:
            pot_region: BGR image of the pot display area.

        Returns:
            Pot amount as integer, or None if not detected.
        """
        return self.extract_amount(pot_region)

    def extract_player_chips(self, chip_region: np.ndarray) -> Optional[int]:
        """
        Extract a player's chip count from their chip display region.

        Args:
            chip_region: BGR image of a player's chip display.

        Returns:
            Chip count as integer, or None if not detected.
        """
        return self.extract_amount(chip_region)

    def extract_bet(self, bet_region: np.ndarray) -> Optional[int]:
        """
        Extract a bet amount from a bet display region.

        Args:
            bet_region: BGR image of a bet amount display.

        Returns:
            Bet amount as integer, or None if not detected.
        """
        return self.extract_amount(bet_region)

    def _parse_amount(self, text: str) -> Optional[int]:
        """
        Parse a numeric amount from OCR text.

        Handles formats like:
        - "1,120" -> 1120
        - "2720" -> 2720
        - "1.5K" -> 1500
        - "2M" -> 2000000

        Args:
            text: OCR output text.

        Returns:
            Parsed integer amount, or None if parsing fails.
        """
        if not text:
            return None

        # Clean the text
        text = text.strip().upper()

        # Remove common OCR mistakes
        text = text.replace('O', '0').replace('I', '1').replace('L', '1')
        text = text.replace('S', '5').replace('B', '8').replace('Z', '2')
        text = text.replace('¢', '0').replace('C', '0')  # ¢ often misread for 0

        # Handle K/M suffixes
        multiplier = 1
        if text.endswith('K'):
            multiplier = 1000
            text = text[:-1]
        elif text.endswith('M'):
            multiplier = 1000000
            text = text[:-1]

        # Remove commas, spaces, and other separators
        text = text.replace(',', '').replace(' ', '').replace('.', '')

        # Extract all digits
        digits = ''.join(c for c in text if c.isdigit())

        if digits:
            try:
                return int(digits) * multiplier
            except ValueError:
                pass

        return None


def extract_chip_amount(region: np.ndarray) -> Optional[int]:
    """
    Convenience function to extract chip amount from a region.

    Args:
        region: BGR image region containing chip amount.

    Returns:
        Integer amount if extracted, None otherwise.
    """
    ocr = ChipOCR()
    return ocr.extract_amount(region)
