"""
Detection functions for hero cards.

Used by video_recorder for auto-recording and video_splitter for segmentation.
"""

import numpy as np

from .card_matcher import match_hero_cards
from ._positions import SEAT_POSITIONS as _SEAT_POSITIONS


def detect_hero_cards(image: np.ndarray) -> bool:
    """
    Check if hero cards are visible at any seat position.

    Args:
        image: The frame image.

    Returns:
        True if both hero cards detected at any position, False otherwise.
    """
    for pos in _SEAT_POSITIONS:
        cards = match_hero_cards(image, pos)
        if cards[0] is not None and cards[1] is not None:
            return True
    return False
