"""
Detection functions for hero cards.

Used by video_recorder for auto-recording and video_splitter for segmentation.
"""

from difflib import SequenceMatcher

import numpy as np

from .card_matcher import match_hero_cards
from ._positions import SEAT_POSITIONS as _SEAT_POSITIONS
from .first_frame import TableInfo
from .fast_ocr import ocr_name_at_position

_NAME_SIMILARITY_THRESHOLD = 0.6


def _is_hero_name(name: str | None, threshold: float = _NAME_SIMILARITY_THRESHOLD) -> bool:
    """
    Check if a name is similar enough to the hero name.

    Args:
        name: Detected name string.
        threshold: Minimum similarity ratio (0-1).

    Returns:
        True if name matches hero name within threshold.
    """
    if name is None:
        return False
    hero_name = TableInfo._HERO_NAME
    ratio = SequenceMatcher(None, name.lower(), hero_name.lower()).ratio()
    return ratio >= threshold


def detect_hero_cards(image: np.ndarray) -> bool:
    """
    Check if hero cards are visible at any seat position with matching hero name.

    Args:
        image: The frame image.

    Returns:
        True if both hero cards detected and hero name matches, False otherwise.
    """
    for pos in _SEAT_POSITIONS:
        cards = match_hero_cards(image, pos)
        if cards[0] is not None and cards[1] is not None:
            name = ocr_name_at_position(image, pos)
            if _is_hero_name(name):
                return True
    return False
