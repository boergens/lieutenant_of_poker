"""
Chip/money detection module for Governor of Poker.

Extracts chip counts, pot amounts, and bet values from game frames.
"""

from collections import deque
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import cv2
import numpy as np

from .fast_ocr import ocr_digits

if TYPE_CHECKING:
    from .table_regions import TableRegionDetector


def _image_fingerprint(image: np.ndarray) -> bytes:
    """Create a fingerprint of an image for cache lookup."""
    small = cv2.resize(image, (16, 8), interpolation=cv2.INTER_AREA)
    if len(small.shape) == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    quantized = (small // 32).astype(np.uint8)
    return quantized.tobytes()


class _Cache:
    """LRU cache for OCR results."""

    def __init__(self, max_size: int = 3):
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


# Module-level caches and stats
_pot_cache = _Cache()
_player_caches: Dict[str, _Cache] = {}
_ocr_calls = 0


def _get_player_cache(position: int) -> _Cache:
    """Get or create cache for a player position (0-4)."""
    key = str(position)
    if key not in _player_caches:
        _player_caches[key] = _Cache()
    return _player_caches[key]


def get_ocr_calls() -> int:
    """Get number of OCR calls (cache misses) since last clear."""
    return _ocr_calls


def clear_caches() -> None:
    """Clear all OCR caches and reset stats."""
    global _pot_cache, _player_caches, _ocr_calls
    _pot_cache = _Cache()
    _player_caches = {}
    _ocr_calls = 0


def _parse_amount(text: str) -> Optional[int]:
    """
    Parse a numeric amount from OCR text.

    Handles formats like "1,120", "2720", "1.5K", "2M".
    """
    if not text:
        return None

    text = text.strip().upper()

    # Common OCR mistakes
    text = text.replace('O', '0').replace('I', '1').replace('L', '1')
    text = text.replace('S', '5').replace('B', '8').replace('Z', '2')
    text = text.replace('¢', '0').replace('C', '0')

    # Handle K/M suffixes
    multiplier = 1
    if text.endswith('K'):
        multiplier = 1000
        text = text[:-1]
    elif text.endswith('M'):
        multiplier = 1000000
        text = text[:-1]

    text = text.replace(',', '').replace(' ', '').replace('.', '')
    digits = ''.join(c for c in text if c.isdigit())

    if digits:
        try:
            return int(digits) * multiplier
        except ValueError:
            pass

    return None


def _ocr_region(region: np.ndarray, category: str = "other") -> Optional[int]:
    """Extract amount from a region using matched filter OCR."""
    global _ocr_calls

    if region is None or region.size == 0:
        return None

    _ocr_calls += 1
    text = ocr_digits(region, category=category)
    return _parse_amount(text)


def extract_pot(
    frame: np.ndarray,
    region_detector: "TableRegionDetector",
) -> Optional[int]:
    """
    Extract pot amount from a game frame.

    Args:
        frame: BGR game frame.
        region_detector: Table region detector for this frame.

    Returns:
        Pot amount as integer, or None if not detected.
    """
    pot_region = region_detector.extract_pot(frame)

    found, cached = _pot_cache.get(pot_region)
    if found:
        return cached

    result = _ocr_region(pot_region, category="pot")
    _pot_cache.put(pot_region, result)
    return result


def extract_player_chips(
    frame: np.ndarray,
    region_detector: "TableRegionDetector",
    position: int,
) -> Optional[int]:
    """
    Extract a player's chip count from a game frame.

    Args:
        frame: BGR game frame.
        region_detector: Table region detector for this frame.
        position: Player seat index (0-4).

    Returns:
        Chip count as integer, or None if not detected.
    """
    chip_region = region_detector.extract_player_chips(frame, position)
    cache = _get_player_cache(position)

    found, cached = cache.get(chip_region)
    if found:
        return cached

    result = _ocr_region(chip_region, category="player")
    cache.put(chip_region, result)
    return result
