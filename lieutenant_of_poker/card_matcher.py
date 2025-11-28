"""
Card matching using a reference library.

Instead of OCR, we compare card images against a library of known cards.
Unknown cards are identified by Claude Code and added to the library.
"""

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .card_detector import Card, Rank, Suit

# Library location (relative to this module)
LIBRARY_DIR = Path(__file__).parent / "card_library"

# Standard size for card comparison (cards are resized to this)
STANDARD_SIZE = (60, 90)

class CardMatcher:
    """Matches card images against a library of known cards."""

    # Maximum difference threshold for a match (lower = stricter)
    MATCH_THRESHOLD = 0.05

    def __init__(self, library_dir: Optional[Path] = None):
        """
        Initialize the card matcher.

        Args:
            library_dir: Path to the card library directory.
        """
        self.library_dir = library_dir or LIBRARY_DIR
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # Cache of loaded reference images: {(rank, suit): [normalized_images]}
        self._library: dict[Tuple[Rank, Suit], list[np.ndarray]] = {}
        self._load_library()

    def _load_library(self) -> None:
        """Load all reference images from the library."""
        self._library.clear()

        for image_path in self.library_dir.glob("*.png"):
            card = self._parse_filename(image_path.name)
            if card is None:
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                continue

            normalized = self._normalize_image(img)
            key = (card.rank, card.suit)

            if key not in self._library:
                self._library[key] = []
            self._library[key].append(normalized)

    def _parse_filename(self, filename: str) -> Optional[Card]:
        """
        Parse card from filename like 'Q_hearts_slot3.png' or 'A_spades.png'.

        Returns Card or None if invalid filename.
        """
        # Remove .png extension
        name = filename.replace(".png", "")

        # Split by underscore
        parts = name.split("_")
        if len(parts) < 2:
            return None

        rank_str = parts[0].upper()
        suit_str = parts[1].lower()

        # Map rank string to Rank enum
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "T": Rank.TEN,
            "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        }

        suit_map = {
            "hearts": Suit.HEARTS, "h": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS, "d": Suit.DIAMONDS,
            "clubs": Suit.CLUBS, "c": Suit.CLUBS,
            "spades": Suit.SPADES, "s": Suit.SPADES,
        }

        rank = rank_map.get(rank_str)
        suit = suit_map.get(suit_str)

        if rank is None or suit is None:
            return None

        return Card(rank=rank, suit=suit)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize an image for comparison."""
        # Resize to standard size
        resized = cv2.resize(img, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
        # Convert to float and normalize
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def _compare_images(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare two normalized images.

        Returns difference score (0 = identical, higher = more different).
        """
        diff = np.abs(img1 - img2)
        return np.mean(diff)

    def match_card(self, card_image: np.ndarray, slot_index: int = 0) -> Optional[Card]:
        """
        Match a card image against the library.

        Args:
            card_image: BGR image of a card.
            slot_index: Which slot this card is from (for naming new library entries).

        Returns:
            Card if matched or identified, None if unable to identify.
        """
        if card_image is None or card_image.size == 0:
            return None

        normalized = self._normalize_image(card_image)

        # Try to find a match in the library
        best_match: Optional[Card] = None
        best_score = float("inf")

        for (rank, suit), ref_images in self._library.items():
            for ref_img in ref_images:
                score = self._compare_images(normalized, ref_img)
                if score < best_score:
                    best_score = score
                    best_match = Card(rank=rank, suit=suit)

        # If we found a good match, return it
        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match found - ask Claude Code to identify
        card = self._identify_with_claude(card_image)

        if card is not None:
            # Save to library for future matches
            self._save_to_library(card_image, card, slot_index)
            # Add to in-memory cache
            key = (card.rank, card.suit)
            if key not in self._library:
                self._library[key] = []
            self._library[key].append(normalized)

        return card

    def _identify_with_claude(self, card_image: np.ndarray) -> Optional[Card]:
        """
        Use Claude Code to identify an unknown card.

        Args:
            card_image: BGR image of the card.

        Returns:
            Card if identified, None otherwise.
        """
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, card_image)

        try:
            # Call Claude Code with the image
            prompt = (
                f"Look at this playing card image: {temp_path}\n"
                "What card is this? Reply with ONLY the rank and suit in this exact format:\n"
                "RANK_SUIT\n"
                "Where RANK is one of: 2,3,4,5,6,7,8,9,10,J,Q,K,A\n"
                "And SUIT is one of: hearts,diamonds,clubs,spades\n"
                "Example: Q_hearts or A_spades\n"
                "Reply with ONLY the card identifier, nothing else."
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

            # Parse response
            response = result.stdout.strip()

            # Look for pattern like "Q_hearts" or "A_spades" in the response
            match = re.search(r"([2-9]|10|[JQKA])_?(hearts|diamonds|clubs|spades)", response, re.IGNORECASE)
            if match:
                rank_str = match.group(1).upper()
                suit_str = match.group(2).lower()

                rank_map = {
                    "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
                    "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
                    "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
                }
                suit_map = {
                    "hearts": Suit.HEARTS, "diamonds": Suit.DIAMONDS,
                    "clubs": Suit.CLUBS, "spades": Suit.SPADES,
                }

                rank = rank_map.get(rank_str)
                suit = suit_map.get(suit_str)

                if rank and suit:
                    return Card(rank=rank, suit=suit)

            return None

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def _save_to_library(self, card_image: np.ndarray, card: Card, slot_index: int) -> None:
        """Save a card image to the library."""
        # Create filename like "Q_hearts_slot3.png"
        rank_str = card.rank.value
        suit_str = card.suit.value

        # Find a unique filename
        base_name = f"{rank_str}_{suit_str}"
        filename = f"{base_name}_slot{slot_index}.png"
        filepath = self.library_dir / filename

        # If file exists, add a counter
        counter = 1
        while filepath.exists():
            filename = f"{base_name}_slot{slot_index}_{counter}.png"
            filepath = self.library_dir / filename
            counter += 1

        cv2.imwrite(str(filepath), card_image)

    def get_library_stats(self) -> dict:
        """Get statistics about the card library."""
        total_images = sum(len(imgs) for imgs in self._library.values())
        unique_cards = len(self._library)

        return {
            "total_images": total_images,
            "unique_cards": unique_cards,
            "cards": [
                f"{card.rank.value} of {card.suit.value}"
                for (rank, suit) in self._library.keys()
                for card in [Card(rank=rank, suit=suit)]
            ],
        }


# Singleton instance for efficiency
_matcher: Optional[CardMatcher] = None


def get_card_matcher() -> CardMatcher:
    """Get the singleton CardMatcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = CardMatcher()
    return _matcher


def match_card(card_image: np.ndarray, slot_index: int = 0) -> Optional[Card]:
    """
    Convenience function to match a card image.

    Args:
        card_image: BGR image of a card.
        slot_index: Which slot this card is from.

    Returns:
        Card if matched or identified, None otherwise.
    """
    return get_card_matcher().match_card(card_image, slot_index)
