"""
Digit matching using reference libraries.

Uses template matching (like card_matcher) to recognize digits 0-9
from chip amount regions. Much faster than OCR for fixed-font game text.
"""

from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from .image_matcher import ImageMatcher


# Library location
LIBRARY_DIR = Path(__file__).parent / "digit_library"


class DigitMatcher(ImageMatcher[str]):
    """Matches digit images against a reference library."""

    MATCH_THRESHOLD = 0.15
    IMAGE_SIZE = (20, 30)  # Digits are taller than wide

    def __init__(self):
        super().__init__(LIBRARY_DIR)

    def _parse_name(self, name: str) -> Optional[str]:
        """Parse digit from filename like '0', '1', etc."""
        if name in "0123456789":
            return name
        return None

    def _value_to_filename(self, value: str) -> str:
        """Convert digit to filename base."""
        return value


# Singleton matcher instance
_matcher: Optional[DigitMatcher] = None


def get_digit_matcher() -> DigitMatcher:
    """Get or create the digit matcher."""
    global _matcher
    if _matcher is None:
        _matcher = DigitMatcher()
    return _matcher


def segment_digits(image: np.ndarray) -> List[np.ndarray]:
    """
    Segment an image containing multiple digits into individual digit images.

    Args:
        image: BGR image containing digits (e.g., "1,234")

    Returns:
        List of individual digit images, left to right.
    """
    if image is None or image.size == 0:
        return []

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to get binary image (assuming light text on dark or vice versa)
    # Try both and pick the one with more contours
    _, thresh_light = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh_dark = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours_light, _ = cv2.findContours(thresh_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use whichever threshold found more contours
    if len(contours_light) >= len(contours_dark):
        contours = contours_light
    else:
        contours = contours_dark

    if not contours:
        return []

    # Get bounding boxes and filter small noise
    min_height = image.shape[0] * 0.3  # Digit should be at least 30% of image height
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= min_height and w > 2:  # Filter noise
            boxes.append((x, y, w, h))

    # Sort left to right
    boxes.sort(key=lambda b: b[0])

    # Extract digit images
    digits = []
    for x, y, w, h in boxes:
        # Add small padding
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        digit_img = image[y1:y2, x1:x2]
        digits.append(digit_img)

    return digits


def match_digits(image: np.ndarray) -> str:
    """
    Match all digits in an image and return the concatenated result.

    Args:
        image: BGR image containing digits.

    Returns:
        String of recognized digits (e.g., "1234").
    """
    digit_images = segment_digits(image)
    if not digit_images:
        return ""

    matcher = get_digit_matcher()
    result = []

    for digit_img in digit_images:
        digit = matcher.match(digit_img)
        if digit is not None:
            result.append(digit)

    return "".join(result)
