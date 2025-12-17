"""
Card matching using separate rank and suit reference libraries.

Cards are represented as 2-character strings like "Ah" (Ace of hearts), "Tc" (Ten of clubs).
Ranks: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
Suits: h (hearts), d (diamonds), c (clubs), s (spades)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .image_matcher import match_image

# Library locations
LIBRARY_DIR = Path(__file__).parent / "card_library"
ASSETS_DIR = Path(__file__).parent / "assets"

# Load active seat templates (card back pattern for non-hero active players)
_ACTIVE_SEAT_LEFT = cv2.imread(str(ASSETS_DIR / "active_seat_left.png"))
_ACTIVE_SEAT_RIGHT = cv2.imread(str(ASSETS_DIR / "active_seat_right.png"))
COMMUNITY_LIBRARY = "community"
HERO_LIBRARY = "hero"

# Hero card offsets relative to player position
HERO_RANK_OFFSET = (-15, -110)  # (dx, dy) from player position to left card rank
HERO_SUIT_OFFSET = (-17, -87)   # (dx, dy) from player position to left card suit
HERO_RANK_SIZE = (17, 22)       # (w, h)
HERO_SUIT_SIZE = (22, 24)       # (w, h)
HERO_CARD_SPACING = 58          # pixels between left and right hero card

# Community card regions (absolute coordinates in frame)
COMMUNITY_LEFT_RANK_REGION = (482, 374, 25, 28)  # x, y, w, h (top of card)
COMMUNITY_LEFT_SUIT_REGION = (482, 404, 25, 32)  # x, y, w, h (below rank)
COMMUNITY_CARD_OFFSET = 79  # pixels between each community card


def _get_library_dirs(library_name: str) -> tuple[Path, Path]:
    """Get rank and suit library directories for a named library."""
    base = LIBRARY_DIR / library_name
    return base / "ranks", base / "suits"


def _extract_region(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a subregion from an image."""
    x, y, w, h = region
    return image[y:y+h, x:x+w]


def _offset_region(region: tuple[int, int, int, int], x_offset: int) -> tuple[int, int, int, int]:
    """Offset a region by a given x amount."""
    x, y, w, h = region
    return (x + x_offset, y, w, h)


def _match_rank(image: np.ndarray, library_name: str) -> tuple[Optional[str], float]:
    """Match a rank image. Returns (rank, score)."""
    rank_dir, _ = _get_library_dirs(library_name)
    return match_image(image, rank_dir)


def _match_suit(image: np.ndarray, library_name: str) -> tuple[Optional[str], float]:
    """Match a suit image. Returns (suit, score)."""
    _, suit_dir = _get_library_dirs(library_name)
    return match_image(image, suit_dir)


def _match_card(rank_img: np.ndarray, suit_img: np.ndarray, library_name: str) -> Optional[str]:
    """Match a card from rank and suit images. Returns string like 'Ah' or None."""
    rank, _ = _match_rank(rank_img, library_name)
    suit, _ = _match_suit(suit_img, library_name)
    if rank and suit:
        return f"{rank}{suit}"
    return None


def match_hero_cards(frame: np.ndarray, hero_position: tuple[int, int]) -> list[Optional[str]]:
    """
    Match both hero cards directly from frame.

    Args:
        frame: The video frame.
        hero_position: (x, y) of hero's player position.

    Returns:
        List of [left_card, right_card], each a string like "Ah" or None.
    """
    px, py = hero_position

    # Left card regions
    left_rank_region = (px + HERO_RANK_OFFSET[0], py + HERO_RANK_OFFSET[1], *HERO_RANK_SIZE)
    left_suit_region = (px + HERO_SUIT_OFFSET[0], py + HERO_SUIT_OFFSET[1], *HERO_SUIT_SIZE)

    # Right card regions
    right_rank_region = _offset_region(left_rank_region, HERO_CARD_SPACING)
    right_suit_region = _offset_region(left_suit_region, HERO_CARD_SPACING)

    left_rank = _extract_region(frame, left_rank_region)
    left_suit = _extract_region(frame, left_suit_region)
    right_rank = _extract_region(frame, right_rank_region)
    right_suit = _extract_region(frame, right_suit_region)

    return [
        _match_card(left_rank, left_suit, HERO_LIBRARY),
        _match_card(right_rank, right_suit, HERO_LIBRARY),
    ]


# Active seat card region (larger than rank/suit, matches template size)
_ACTIVE_CARD_SIZE = (30, 50)  # w, h - matches active_seat_*.png templates


def is_seat_active(frame: np.ndarray, pos: tuple[int, int], threshold: float = 0.7) -> bool:
    """
    Check if a seat has an active player.

    A seat is active if it shows either hero cards or the card back pattern.

    Args:
        frame: BGR game frame.
        pos: (x, y) seat position coordinates.
        threshold: Minimum match score for template matching.

    Returns:
        True if seat has an active player.
    """
    # First check for hero cards
    cards = match_hero_cards(frame, pos)
    if any(c is not None for c in cards):
        return True

    # Check for card back pattern
    if _ACTIVE_SEAT_LEFT is None or _ACTIVE_SEAT_RIGHT is None:
        return False

    px, py = pos
    left_region = (px + HERO_RANK_OFFSET[0], py + HERO_RANK_OFFSET[1], *_ACTIVE_CARD_SIZE)
    right_region = _offset_region(left_region, HERO_CARD_SPACING)

    left_img = _extract_region(frame, left_region)
    right_img = _extract_region(frame, right_region)

    # Match left template
    if left_img.shape[:2] == _ACTIVE_SEAT_LEFT.shape[:2]:
        result = cv2.matchTemplate(left_img, _ACTIVE_SEAT_LEFT, cv2.TM_CCOEFF_NORMED)
        if result[0, 0] >= threshold:
            return True

    # Match right template
    if right_img.shape[:2] == _ACTIVE_SEAT_RIGHT.shape[:2]:
        result = cv2.matchTemplate(right_img, _ACTIVE_SEAT_RIGHT, cv2.TM_CCOEFF_NORMED)
        if result[0, 0] >= threshold:
            return True

    return False


def match_community_cards(frame: np.ndarray, num_cards: int = 5) -> list[Optional[str]]:
    """
    Match community cards directly from frame.

    Returns:
        List of cards, each a string like "Ah" or None.
    """
    cards = []
    for i in range(num_cards):
        offset = i * COMMUNITY_CARD_OFFSET
        rank_img = _extract_region(frame, _offset_region(COMMUNITY_LEFT_RANK_REGION, offset))
        suit_img = _extract_region(frame, _offset_region(COMMUNITY_LEFT_SUIT_REGION, offset))
        cards.append(_match_card(rank_img, suit_img, COMMUNITY_LIBRARY))
    return cards


# For diagnostic display
class HeroCardRegions:
    """Extracted rank and suit regions for both hero cards."""
    def __init__(self, left_rank: np.ndarray, left_suit: np.ndarray,
                 right_rank: np.ndarray, right_suit: np.ndarray):
        self.left_rank = left_rank
        self.left_suit = left_suit
        self.right_rank = right_rank
        self.right_suit = right_suit


class CommunityCardRegions:
    """Extracted rank and suit regions for all community cards."""
    def __init__(self, cards: list[tuple[np.ndarray, np.ndarray]]):
        self.cards = cards

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.cards[index]

    def __len__(self) -> int:
        return len(self.cards)


def extract_hero_card_regions(frame: np.ndarray) -> HeroCardRegions:
    """Extract rank and suit image regions for both hero cards."""
    return HeroCardRegions(
        left_rank=_extract_region(frame, HERO_LEFT_RANK_REGION),
        left_suit=_extract_region(frame, HERO_LEFT_SUIT_REGION),
        right_rank=_extract_region(frame, _offset_region(HERO_LEFT_RANK_REGION, HERO_CARD_OFFSET)),
        right_suit=_extract_region(frame, _offset_region(HERO_LEFT_SUIT_REGION, HERO_CARD_OFFSET)),
    )


def extract_community_card_regions(frame: np.ndarray, num_cards: int = 5) -> CommunityCardRegions:
    """Extract rank and suit image regions for community cards."""
    cards = []
    for i in range(num_cards):
        offset = i * COMMUNITY_CARD_OFFSET
        cards.append((
            _extract_region(frame, _offset_region(COMMUNITY_LEFT_RANK_REGION, offset)),
            _extract_region(frame, _offset_region(COMMUNITY_LEFT_SUIT_REGION, offset)),
        ))
    return CommunityCardRegions(cards)


def clear_library() -> None:
    """Clear all cached reference images from card libraries."""
    import sys
    count = 0
    for png_file in LIBRARY_DIR.rglob("*.png"):
        png_file.unlink()
        count += 1
    print(f"Cleared {count} card library images", file=sys.stderr)
