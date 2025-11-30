"""
Card matching using separate rank and suit reference libraries.

Instead of OCR, we compare card images against libraries of known ranks and suits.
Unknown images are identified by Claude Code and added to the libraries.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .card_detector import Card, Rank, Suit
from .image_matcher import ImageMatcher

# Library locations
LIBRARY_DIR = Path(__file__).parent / "card_library"

# Library names for different card positions
COMMUNITY_LIBRARY = "community"
HERO_LEFT_LIBRARY = "hero_left"
HERO_RIGHT_LIBRARY = "hero_right"


def get_library_dirs(library_name: str) -> tuple[Path, Path]:
    """Get rank and suit library directories for a named library."""
    base = LIBRARY_DIR / library_name
    return base / "ranks", base / "suits"


# Crop regions within a community card slot (at ~103x146 slot size)
RANK_REGION = (10, 15, 55, 50)  # x, y, w, h
SUIT_REGION = (30, 75, 60, 55)  # x, y, w, h

# Hero card subregions (relative to hero_cards_region)
# Calibrated at 420x220 hero region size - will be scaled proportionally
HERO_CALIBRATION_SIZE = (420, 220)  # width, height at calibration time
HERO_LEFT_RANK_REGION = (116, 21, 85, 92)   # x, y, w, h
HERO_LEFT_SUIT_REGION = (168, 125, 92, 71)  # x, y, w, h
HERO_RIGHT_RANK_REGION = (280, 15, 63, 87)  # x, y, w, h
HERO_RIGHT_SUIT_REGION = (302, 110, 86, 83) # x, y, w, h


def _scale_hero_region(region: tuple[int, int, int, int], hero_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Scale a hero subregion to match actual hero region size."""
    cal_w, cal_h = HERO_CALIBRATION_SIZE
    actual_w, actual_h = hero_size
    scale_x = actual_w / cal_w
    scale_y = actual_h / cal_h
    x, y, w, h = region
    return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))


class RankMatcher(ImageMatcher[Rank]):
    """Matches rank images against a library."""

    MATCH_THRESHOLD = 0.08
    IMAGE_SIZE = (40, 40)

    def __init__(self, library_name: str = COMMUNITY_LIBRARY, library_dir: Optional[Path] = None):
        if library_dir is not None:
            self._library_dir = library_dir
        else:
            rank_dir, _ = get_library_dirs(library_name)
            self._library_dir = rank_dir
        super().__init__(self._library_dir)

    def _parse_name(self, name: str) -> Optional[Rank]:
        """Parse rank from base name like 'Q' or '10'."""
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        }
        return rank_map.get(name.upper())

    def _value_to_filename(self, value: Rank) -> str:
        """Convert rank to filename base."""
        return value.value

    def _get_claude_prompt(self, image_path: str) -> str:
        """Get prompt for Claude to identify a rank."""
        return (
            f"Look at this playing card rank image: {image_path}\n"
            "What rank/number is shown? Reply with ONLY the rank:\n"
            "One of: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A\n"
            "Reply with ONLY the rank character(s), nothing else."
        )

    def _parse_claude_response(self, response: str) -> Optional[Rank]:
        """Parse Claude's response into a rank."""
        response = response.upper()
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN,
            "K": Rank.KING, "A": Rank.ACE,
        }
        for key, rank in rank_map.items():
            if key in response:
                return rank
        return None


class SuitMatcher(ImageMatcher[Suit]):
    """Matches suit images against a library."""

    MATCH_THRESHOLD = 0.08
    IMAGE_SIZE = (40, 40)

    def __init__(self, library_name: str = COMMUNITY_LIBRARY, library_dir: Optional[Path] = None):
        if library_dir is not None:
            self._library_dir = library_dir
        else:
            _, suit_dir = get_library_dirs(library_name)
            self._library_dir = suit_dir
        super().__init__(self._library_dir)

    def _parse_name(self, name: str) -> Optional[Suit]:
        """Parse suit from base name like 'hearts'."""
        suit_map = {
            "hearts": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS,
            "clubs": Suit.CLUBS,
            "spades": Suit.SPADES,
        }
        return suit_map.get(name.lower())

    def _value_to_filename(self, value: Suit) -> str:
        """Convert suit to filename base."""
        return value.value

    def _get_claude_prompt(self, image_path: str) -> str:
        """Get prompt for Claude to identify a suit."""
        return (
            f"Look at this playing card suit symbol: {image_path}\n"
            "What suit is shown? Reply with ONLY one of:\n"
            "hearts, diamonds, clubs, spades\n"
            "Reply with ONLY the suit name, nothing else."
        )

    def _parse_claude_response(self, response: str) -> Optional[Suit]:
        """Parse Claude's response into a suit."""
        response = response.lower()
        suit_map = {
            "hearts": Suit.HEARTS, "diamonds": Suit.DIAMONDS,
            "clubs": Suit.CLUBS, "spades": Suit.SPADES,
        }
        for key, suit in suit_map.items():
            if key in response:
                return suit
        return None


class CardMatcher:
    """Matches card images using separate rank and suit matchers."""

    def __init__(self, library_name: str = COMMUNITY_LIBRARY):
        self.library_name = library_name
        self.rank_matcher = RankMatcher(library_name=library_name)
        self.suit_matcher = SuitMatcher(library_name=library_name)

    def extract_rank_region(self, slot_image: np.ndarray) -> np.ndarray:
        """Extract the rank region from a card slot image."""
        x, y, w, h = RANK_REGION
        return slot_image[y:y+h, x:x+w]

    def extract_suit_region(self, slot_image: np.ndarray) -> np.ndarray:
        """Extract the suit region from a card slot image."""
        x, y, w, h = SUIT_REGION
        return slot_image[y:y+h, x:x+w]

    def match_card(self, card_image: np.ndarray, slot_index: int = 0) -> Optional[Card]:
        """
        Match a card image by separately matching rank and suit.

        Args:
            card_image: BGR image of a card slot.
            slot_index: Ignored (kept for API compatibility).

        Returns:
            Card if both rank and suit matched, None otherwise.
        """
        if card_image is None or card_image.size == 0:
            return None

        # Extract regions
        rank_region = self.extract_rank_region(card_image)
        suit_region = self.extract_suit_region(card_image)

        # Match separately
        rank = self.rank_matcher.match(rank_region)
        suit = self.suit_matcher.match(suit_region)

        if rank is not None and suit is not None:
            return Card(rank=rank, suit=suit)
        return None

    def get_library_stats(self) -> dict:
        """Get statistics about the card libraries."""
        return {
            "library_name": self.library_name,
            "ranks": self.rank_matcher.get_library_size(),
            "suits": self.suit_matcher.get_library_size(),
            "total_possible": 13 + 4,  # 13 ranks + 4 suits
        }


# Singleton instances by library name
_matchers: dict[str, CardMatcher] = {}


def get_card_matcher(library_name: str = COMMUNITY_LIBRARY) -> CardMatcher:
    """Get the CardMatcher instance for a specific library."""
    global _matchers
    if library_name not in _matchers:
        _matchers[library_name] = CardMatcher(library_name=library_name)
    return _matchers[library_name]


def match_card(card_image: np.ndarray, slot_index: int = 0) -> Optional[Card]:
    """Convenience function to match a card image using the appropriate library."""
    # Determine library based on slot index
    # Slots 0-4 are community cards, 5 is hero left, 6 is hero right
    if slot_index == 5:
        library_name = HERO_LEFT_LIBRARY
    elif slot_index == 6:
        library_name = HERO_RIGHT_LIBRARY
    else:
        library_name = COMMUNITY_LIBRARY
    return get_card_matcher(library_name).match_card(card_image, slot_index)


def _extract_region(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a subregion from an image."""
    x, y, w, h = region
    return image[y:y+h, x:x+w]


def match_hero_cards(hero_region: np.ndarray) -> list[Optional[Card]]:
    """
    Match both hero cards from the full hero_cards_region.

    Args:
        hero_region: The full hero cards region image (both cards visible).

    Returns:
        List of [left_card, right_card], each may be None if not detected.
    """
    from .card_detector import CardDetector

    results: list[Optional[Card]] = [None, None]
    detector = CardDetector(use_library=False)  # Just for background check

    # Get actual hero region size for scaling
    h, w = hero_region.shape[:2]
    hero_size = (w, h)

    # Scale regions to match actual hero region size
    left_rank_region = _scale_hero_region(HERO_LEFT_RANK_REGION, hero_size)
    left_suit_region = _scale_hero_region(HERO_LEFT_SUIT_REGION, hero_size)
    right_rank_region = _scale_hero_region(HERO_RIGHT_RANK_REGION, hero_size)
    right_suit_region = _scale_hero_region(HERO_RIGHT_SUIT_REGION, hero_size)

    # Get matchers for hero cards
    left_matcher = get_card_matcher(HERO_LEFT_LIBRARY)
    right_matcher = get_card_matcher(HERO_RIGHT_LIBRARY)

    # Extract and match LEFT card
    left_rank_img = _extract_region(hero_region, left_rank_region)
    left_suit_img = _extract_region(hero_region, left_suit_region)

    # Skip if region matches table background (no card present)
    if not detector.is_empty_slot(left_rank_img):
        left_rank = left_matcher.rank_matcher.match(left_rank_img)
        left_suit = left_matcher.suit_matcher.match(left_suit_img)

        if left_rank and left_suit:
            results[0] = Card(rank=left_rank, suit=left_suit)

    # Extract and match RIGHT card
    right_rank_img = _extract_region(hero_region, right_rank_region)
    right_suit_img = _extract_region(hero_region, right_suit_region)

    # Skip if region matches table background (no card present)
    if not detector.is_empty_slot(right_rank_img):
        right_rank = right_matcher.rank_matcher.match(right_rank_img)
        right_suit = right_matcher.suit_matcher.match(right_suit_img)

        if right_rank and right_suit:
            results[1] = Card(rank=right_rank, suit=right_suit)

    return results
