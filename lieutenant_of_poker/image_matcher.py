"""
Generic image matching library.

Provides a base class for matching images against a library of known references.
Used by card_matcher and action_matcher for their specific domains.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

import cv2
import numpy as np

# Generic type for the matched value (Rank, Suit, PlayerAction, etc.)
T = TypeVar("T")

# Global flag that gets set when an unmatched image is saved
_unmatched_saved: bool = False


def unmatched_was_saved() -> bool:
    """Check if an unmatched image was saved since last reset."""
    return _unmatched_saved


def reset_unmatched_flag() -> None:
    """Reset the unmatched saved flag."""
    global _unmatched_saved
    _unmatched_saved = False


# Keep old names as aliases for compatibility
claude_was_called = unmatched_was_saved
reset_claude_flag = reset_unmatched_flag


class ImageMatcher(ABC, Generic[T]):
    """
    Base class for matching images against a reference library.

    Subclasses define how to parse filenames and convert between enum values
    and string names. Unmatched images are saved to an 'unmatched' subfolder.
    """

    # Override in subclass
    MATCH_THRESHOLD: float = 0.10
    IMAGE_SIZE: tuple[int, int] = (40, 40)

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._library: dict[T, list[np.ndarray]] = {}
        self._unmatched_images: list[np.ndarray] = []
        self._load_library()
        self._load_unmatched()

    def _load_library(self) -> None:
        """Load all reference images from the library directory."""
        self._library.clear()
        for image_path in self.library_dir.glob("*.png"):
            # Parse filename: NAME_N.png -> NAME
            name = image_path.stem  # filename without .png
            parts = name.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue  # Invalid format, skip
            base_name = parts[0]

            value = self._parse_name(base_name)
            if value is None:
                continue
            img = cv2.imread(str(image_path))
            if img is not None:
                if value not in self._library:
                    self._library[value] = []
                self._library[value].append(self._normalize(img))

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for comparison."""
        resized = cv2.resize(img, self.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _load_unmatched(self) -> None:
        """Load existing unmatched images to avoid saving duplicates."""
        self._unmatched_images.clear()
        unmatched_dir = self.library_dir / "unmatched"
        if not unmatched_dir.exists():
            return
        for image_path in unmatched_dir.glob("*.png"):
            img = cv2.imread(str(image_path))
            if img is not None:
                self._unmatched_images.append(self._normalize(img))

    def match(self, image: np.ndarray) -> Optional[T]:
        """
        Match an image against the library.

        Returns the matched value, or None if no match found.
        Unmatched images are saved to the 'unmatched' subfolder for manual review.
        """
        if image is None or image.size == 0:
            return None

        normalized = self._normalize(image)
        best_match: Optional[T] = None
        best_score = float("inf")

        for value, ref_images in self._library.items():
            for ref_img in ref_images:
                score = float(np.mean(np.abs(normalized - ref_img)))
                if score < best_score:
                    best_score = score
                    best_match = value

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - save to unmatched folder
        self._save_unmatched(image)
        return None

    def _save_unmatched(self, image: np.ndarray) -> None:
        """Save an unmatched image to the 'unmatched' subfolder if not a duplicate."""
        global _unmatched_saved
        _unmatched_saved = True

        normalized = self._normalize(image)

        # Check if this matches an existing unmatched image
        for unmatched_img in self._unmatched_images:
            if float(np.mean(np.abs(normalized - unmatched_img))) < self.MATCH_THRESHOLD:
                return  # Already have this one, skip

        unmatched_dir = self.library_dir / "unmatched"
        unmatched_dir.mkdir(parents=True, exist_ok=True)

        # Find next available index
        existing = list(unmatched_dir.glob("*.png"))
        next_index = len(existing)

        filepath = unmatched_dir / f"unmatched_{next_index}.png"
        cv2.imwrite(str(filepath), image)

        # Add to cache to prevent future duplicates
        self._unmatched_images.append(normalized)

    def _save_to_library(self, image: np.ndarray, value: T, variant: int) -> None:
        """Save an image to the library with variant index."""
        name = self._value_to_filename(value)
        filename = f"{name}_{variant}.png"
        filepath = self.library_dir / filename
        if not filepath.exists():
            cv2.imwrite(str(filepath), image)

    def get_library_size(self) -> int:
        """Return total number of images in library."""
        return sum(len(images) for images in self._library.values())

    @abstractmethod
    def _parse_name(self, name: str) -> Optional[T]:
        """Parse a value from a base name (e.g., 'FOLD', 'Q', 'hearts')."""
        pass

    @abstractmethod
    def _value_to_filename(self, value: T) -> str:
        """Convert a value to a filename base (without variant suffix or extension)."""
        pass

class ImageMatcherWithNone(ImageMatcher[T]):
    """
    Image matcher that also tracks "no match" images.

    Useful for cases like action detection where an empty/no-action image
    should be recognized and not repeatedly flagged as unmatched.
    """

    def __init__(self, library_dir: Path):
        self._none_images: list[np.ndarray] = list()
        super().__init__(library_dir)

    def _load_library(self) -> None:
        """Load all reference images including NONE images."""
        self._library.clear()
        self._none_images.clear()
        for image_path in self.library_dir.glob("*.png"):
            # Parse filename: NAME_N.png -> NAME
            name = image_path.stem  # filename without .png
            parts = name.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue  # Invalid format, skip
            base_name = parts[0]

            # Load "no match" images
            if base_name.upper() == "NONE":
                img = cv2.imread(str(image_path))
                if img is not None:
                    self._none_images.append(self._normalize(img))
                continue

            value = self._parse_name(base_name)
            if value is None:
                continue
            img = cv2.imread(str(image_path))
            if img is not None:
                if value not in self._library:
                    self._library[value] = []
                self._library[value].append(self._normalize(img))

    def match(self, image: np.ndarray) -> Optional[T]:
        """Match an image, checking for known "no match" images first."""
        if image is None or image.size == 0:
            return None

        normalized = self._normalize(image)

        # Check if this matches a known "no match" image
        for none_img in self._none_images:
            if float(np.mean(np.abs(normalized - none_img))) < self.MATCH_THRESHOLD:
                return None

        # Check against main library
        best_match: Optional[T] = None
        best_score = float("inf")

        for value, ref_images in self._library.items():
            for ref_img in ref_images:
                score = float(np.mean(np.abs(normalized - ref_img)))
                if score < best_score:
                    best_score = score
                    best_match = value

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - save to unmatched folder
        self._save_unmatched(image)
        return None

    def _save_none_image(self, image: np.ndarray, variant: int) -> None:
        """Save a 'no match' image to the library."""
        filename = f"NONE_{variant}.png"
        filepath = self.library_dir / filename
        if not filepath.exists():
            cv2.imwrite(str(filepath), image)
