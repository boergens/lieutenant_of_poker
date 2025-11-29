"""
Card matching using separate rank and suit reference libraries.

Instead of OCR, we compare card images against libraries of known ranks and suits.
Unknown images are identified by Claude Code and added to the libraries.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .card_detector import Card, Rank, Suit

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

# Standard sizes for comparison
RANK_SIZE = (40, 40)
SUIT_SIZE = (40, 40)

# Crop regions within a card slot (at ~103x146 slot size)
RANK_REGION = (10, 15, 55, 55)  # x, y, w, h
SUIT_REGION = (30, 75, 60, 55)  # x, y, w, h


class RankMatcher:
    """Matches rank images against a library."""

    MATCH_THRESHOLD = 0.08

    def __init__(self, library_name: str = COMMUNITY_LIBRARY, library_dir: Optional[Path] = None):
        if library_dir is not None:
            self.library_dir = library_dir
        else:
            rank_dir, _ = get_library_dirs(library_name)
            self.library_dir = rank_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._library: dict[Rank, np.ndarray] = {}
        self._load_library()

    def _load_library(self) -> None:
        """Load all rank reference images."""
        self._library.clear()
        for image_path in self.library_dir.glob("*.png"):
            rank = self._parse_filename(image_path.name)
            if rank is None:
                continue
            img = cv2.imread(str(image_path))
            if img is not None:
                self._library[rank] = self._normalize(img)

    def _parse_filename(self, filename: str) -> Optional[Rank]:
        """Parse rank from filename like 'Q.png' or '10.png'."""
        name = filename.replace(".png", "").upper()
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "T": Rank.TEN,
            "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        }
        return rank_map.get(name)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for comparison."""
        resized = cv2.resize(img, RANK_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _compare(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare two normalized images. Returns difference score."""
        return np.mean(np.abs(img1 - img2))

    def match(self, rank_image: np.ndarray) -> Optional[Rank]:
        """Match a rank image against the library."""
        if rank_image is None or rank_image.size == 0:
            return None

        normalized = self._normalize(rank_image)
        best_match: Optional[Rank] = None
        best_score = float("inf")

        for rank, ref_img in self._library.items():
            score = self._compare(normalized, ref_img)
            if score < best_score:
                best_score = score
                best_match = rank

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - ask Claude
        rank = self._identify_with_claude(rank_image)
        if rank is not None:
            self._save_to_library(rank_image, rank)
            self._library[rank] = normalized
        return rank

    def _identify_with_claude(self, image: np.ndarray) -> Optional[Rank]:
        """Use Claude Code to identify an unknown rank."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        try:
            prompt = (
                f"Look at this playing card rank image: {temp_path}\n"
                "What rank/number is shown? Reply with ONLY the rank:\n"
                "One of: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A\n"
                "Reply with ONLY the rank character(s), nothing else."
            )

            claude_path = Path.home() / ".local" / "bin" / "claude"
            result = subprocess.run(
                [str(claude_path), "-p", prompt, "--allowedTools", "Read"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            response = result.stdout.strip().upper()
            rank_map = {
                "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
                "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
                "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN,
                "K": Rank.KING, "A": Rank.ACE,
            }

            # Try to find a rank in the response
            for key, rank in rank_map.items():
                if key in response:
                    return rank
            return None

        except Exception:
            return None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _save_to_library(self, image: np.ndarray, rank: Rank) -> None:
        """Save a rank image to the library."""
        filename = f"{rank.value}.png"
        filepath = self.library_dir / filename
        if not filepath.exists():
            cv2.imwrite(str(filepath), image)

    def get_library_size(self) -> int:
        return len(self._library)


class SuitMatcher:
    """Matches suit images against a library."""

    MATCH_THRESHOLD = 0.08

    def __init__(self, library_name: str = COMMUNITY_LIBRARY, library_dir: Optional[Path] = None):
        if library_dir is not None:
            self.library_dir = library_dir
        else:
            _, suit_dir = get_library_dirs(library_name)
            self.library_dir = suit_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._library: dict[Suit, np.ndarray] = {}
        self._load_library()

    def _load_library(self) -> None:
        """Load all suit reference images."""
        self._library.clear()
        for image_path in self.library_dir.glob("*.png"):
            suit = self._parse_filename(image_path.name)
            if suit is None:
                continue
            img = cv2.imread(str(image_path))
            if img is not None:
                self._library[suit] = self._normalize(img)

    def _parse_filename(self, filename: str) -> Optional[Suit]:
        """Parse suit from filename like 'hearts.png'."""
        name = filename.replace(".png", "").lower()
        suit_map = {
            "hearts": Suit.HEARTS, "h": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS, "d": Suit.DIAMONDS,
            "clubs": Suit.CLUBS, "c": Suit.CLUBS,
            "spades": Suit.SPADES, "s": Suit.SPADES,
        }
        return suit_map.get(name)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for comparison."""
        resized = cv2.resize(img, SUIT_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _compare(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare two normalized images. Returns difference score."""
        return np.mean(np.abs(img1 - img2))

    def match(self, suit_image: np.ndarray) -> Optional[Suit]:
        """Match a suit image against the library."""
        if suit_image is None or suit_image.size == 0:
            return None

        normalized = self._normalize(suit_image)
        best_match: Optional[Suit] = None
        best_score = float("inf")

        for suit, ref_img in self._library.items():
            score = self._compare(normalized, ref_img)
            if score < best_score:
                best_score = score
                best_match = suit

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - ask Claude
        suit = self._identify_with_claude(suit_image)
        if suit is not None:
            self._save_to_library(suit_image, suit)
            self._library[suit] = normalized
        return suit

    def _identify_with_claude(self, image: np.ndarray) -> Optional[Suit]:
        """Use Claude Code to identify an unknown suit."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        try:
            prompt = (
                f"Look at this playing card suit symbol: {temp_path}\n"
                "What suit is shown? Reply with ONLY one of:\n"
                "hearts, diamonds, clubs, spades\n"
                "Reply with ONLY the suit name, nothing else."
            )

            claude_path = Path.home() / ".local" / "bin" / "claude"
            result = subprocess.run(
                [str(claude_path), "-p", prompt, "--allowedTools", "Read"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            response = result.stdout.strip().lower()
            suit_map = {
                "hearts": Suit.HEARTS, "diamonds": Suit.DIAMONDS,
                "clubs": Suit.CLUBS, "spades": Suit.SPADES,
            }

            for key, suit in suit_map.items():
                if key in response:
                    return suit
            return None

        except Exception:
            return None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _save_to_library(self, image: np.ndarray, suit: Suit) -> None:
        """Save a suit image to the library."""
        filename = f"{suit.value}.png"
        filepath = self.library_dir / filename
        if not filepath.exists():
            cv2.imwrite(str(filepath), image)

    def get_library_size(self) -> int:
        return len(self._library)


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
