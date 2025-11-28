"""
Chip/money OCR module for Governor of Poker.

Extracts chip counts, pot amounts, and bet values from game UI regions.
"""

import re
from typing import Optional

import cv2
import numpy as np
import pytesseract


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

        # Preprocess the image
        processed = self._preprocess(region)

        # Try multiple OCR configurations
        for config in self._get_ocr_configs():
            text = pytesseract.image_to_string(processed, config=config).strip()
            amount = self._parse_amount(text)
            if amount is not None:
                return amount

        # Try with inverted colors
        inverted = cv2.bitwise_not(processed)
        for config in self._get_ocr_configs():
            text = pytesseract.image_to_string(inverted, config=config).strip()
            amount = self._parse_amount(text)
            if amount is not None:
                return amount

        return None

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

    def _preprocess(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for OCR.

        Args:
            region: BGR image region.

        Returns:
            Preprocessed grayscale image.
        """
        # Scale up for better OCR
        scaled = cv2.resize(region, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Extract bright text (white/light colored numbers)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Invert so text is black on white background (better for OCR)
        inverted = cv2.bitwise_not(bright)

        return inverted

    def _get_ocr_configs(self) -> list:
        """Get list of OCR configurations to try."""
        return [
            '--psm 7 -c tessedit_char_whitelist=0123456789,.',  # Single line
            '--psm 8 -c tessedit_char_whitelist=0123456789,.',  # Single word
            '--psm 6 -c tessedit_char_whitelist=0123456789,.',  # Block of text
            '--psm 13 -c tessedit_char_whitelist=0123456789,.',  # Raw line
        ]

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
