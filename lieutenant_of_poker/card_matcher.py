"""
Card matching using separate rank and suit reference libraries.

Cards are represented as 2-character strings like "Ah" (Ace of hearts), "Tc" (Ten of clubs).
Ranks: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
Suits: h (hearts), d (diamonds), c (clubs), s (spades)
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .image_matcher import match_image

# Expected frame resolution (all coordinates are absolute for this resolution)
FRAME_WIDTH = 1342
FRAME_HEIGHT = 960

# Library locations
LIBRARY_DIR = Path(__file__).parent / "card_library"
COMMUNITY_LIBRARY = "community"
HERO_LIBRARY = "hero"

# Hero card regions (absolute coordinates in frame)
HERO_LEFT_RANK_REGION = (831, 600, 17, 22)   # x, y, w, h (top of card)
HERO_LEFT_SUIT_REGION = (829, 623, 22, 24)   # x, y, w, h (below rank)
HERO_CARD_OFFSET = 58  # pixels between left and right hero card

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


def _match_rank(image: np.ndarray, library_name: str) -> Optional[str]:
    """Match a rank image, handling 10->T conversion."""
    rank_dir, _ = _get_library_dirs(library_name)
    result = match_image(image, rank_dir)
    if result == "10":
        return "T"
    return result


def _match_suit(image: np.ndarray, library_name: str) -> Optional[str]:
    """Match a suit image."""
    _, suit_dir = _get_library_dirs(library_name)
    return match_image(image, suit_dir)


def _match_card(rank_img: np.ndarray, suit_img: np.ndarray, library_name: str) -> Optional[str]:
    """Match a card from rank and suit images. Returns string like 'Ah' or None."""
    rank = _match_rank(rank_img, library_name)
    suit = _match_suit(suit_img, library_name)
    if rank and suit:
        return f"{rank}{suit}"
    return None


def match_hero_cards(frame: np.ndarray) -> list[Optional[str]]:
    """
    Match both hero cards directly from frame.

    Returns:
        List of [left_card, right_card], each a string like "Ah" or None.
    """
    left_rank = _extract_region(frame, HERO_LEFT_RANK_REGION)
    left_suit = _extract_region(frame, HERO_LEFT_SUIT_REGION)
    right_rank = _extract_region(frame, _offset_region(HERO_LEFT_RANK_REGION, HERO_CARD_OFFSET))
    right_suit = _extract_region(frame, _offset_region(HERO_LEFT_SUIT_REGION, HERO_CARD_OFFSET))

    return [
        _match_card(left_rank, left_suit, HERO_LIBRARY),
        _match_card(right_rank, right_suit, HERO_LIBRARY),
    ]


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
