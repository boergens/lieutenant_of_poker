"""
Generic image matching library.

Provides a base class for matching images against a library of known references.
Used by card_matcher and action_matcher for their specific domains.
"""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Generic, Optional, TypeVar

import cv2
import numpy as np

# Generic type for the matched value (Rank, Suit, PlayerAction, etc.)
T = TypeVar("T")

# Global flag that gets set when Claude is invoked
_claude_was_called: bool = False


def claude_was_called() -> bool:
    """Check if Claude was called since last reset."""
    return _claude_was_called


def reset_claude_flag() -> None:
    """Reset the Claude called flag."""
    global _claude_was_called
    _claude_was_called = False


def _notify_claude_called() -> None:
    """Internal: mark that Claude was called."""
    global _claude_was_called
    _claude_was_called = True


class ImageMatcher(ABC, Generic[T]):
    """
    Base class for matching images against a reference library.

    Subclasses define how to parse filenames, identify unknown images via Claude,
    and convert between enum values and string names.
    """

    # Override in subclass
    MATCH_THRESHOLD: float = 0.10
    IMAGE_SIZE: tuple[int, int] = (40, 40)

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._library: dict[T, list[np.ndarray]] = {}
        self._load_library()

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

    def _compare(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare two normalized images. Returns difference score (lower = more similar)."""
        return float(np.mean(np.abs(img1 - img2)))

    def match(self, image: np.ndarray) -> Optional[T]:
        """
        Match an image against the library.

        Returns the matched value, or None if no match found.
        Unknown images are identified via Claude and added to the library.
        """
        if image is None or image.size == 0:
            return None

        normalized = self._normalize(image)
        best_match: Optional[T] = None
        best_score = float("inf")

        for value, ref_images in self._library.items():
            for ref_img in ref_images:
                score = self._compare(normalized, ref_img)
                if score < best_score:
                    best_score = score
                    best_match = value

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - ask Claude
        result = self._identify_with_claude(image)
        if result is not None:
            if result not in self._library:
                self._library[result] = []
            self._save_to_library(image, result, len(self._library[result]))
            self._library[result].append(normalized)
        return result

    def _identify_with_claude(self, image: np.ndarray) -> Optional[T]:
        """Use Claude Code to identify an unknown image."""
        _notify_claude_called()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        try:
            prompt = self._get_claude_prompt(temp_path)
            claude_path = Path.home() / ".local" / "bin" / "claude"
            result = subprocess.run(
                [str(claude_path), "-p", prompt, "--allowedTools", "Read"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            return self._parse_claude_response(result.stdout.strip())

        except Exception:
            return None
        finally:
            Path(temp_path).unlink(missing_ok=True)

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

    @abstractmethod
    def _get_claude_prompt(self, image_path: str) -> str:
        """Get the prompt to send to Claude for identifying an unknown image."""
        pass

    @abstractmethod
    def _parse_claude_response(self, response: str) -> Optional[T]:
        """Parse Claude's response into a value."""
        pass


class ImageMatcherWithNone(ImageMatcher[T]):
    """
    Image matcher that also tracks "no match" images.

    Useful for cases like action detection where an empty/no-action image
    should be recognized and not repeatedly sent to Claude.
    """

    def __init__(self, library_dir: Path):
        self._none_images: list[np.ndarray] = []
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
            if self._compare(normalized, none_img) < self.MATCH_THRESHOLD:
                return None

        # Check against main library
        best_match: Optional[T] = None
        best_score = float("inf")

        for value, ref_images in self._library.items():
            for ref_img in ref_images:
                score = self._compare(normalized, ref_img)
                if score < best_score:
                    best_score = score
                    best_match = value

        if best_match is not None and best_score < self.MATCH_THRESHOLD:
            return best_match

        # No match - ask Claude
        result = self._identify_with_claude(image)
        if result is None:
            # Claude said no match - save as NONE
            self._save_none_image(image, len(self._none_images))
            self._none_images.append(normalized)
        else:
            if result not in self._library:
                self._library[result] = []
            self._save_to_library(image, result, len(self._library[result]))
            self._library[result].append(normalized)
        return result

    def _save_none_image(self, image: np.ndarray, variant: int) -> None:
        """Save a 'no match' image to the library."""
        filename = f"NONE_{variant}.png"
        filepath = self.library_dir / filename
        if not filepath.exists():
            cv2.imwrite(str(filepath), image)
