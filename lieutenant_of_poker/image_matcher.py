"""
Simple image matching against a library of reference images.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Cache for loaded libraries: {library_dir: {value: [normalized_images]}}
_library_cache: dict[Path, dict[str, list[np.ndarray]]] = {}

# Global flag for tracking unmatched images
_unmatched_saved: bool = False


def unmatched_was_saved() -> bool:
    """Check if an unmatched image was saved since last reset."""
    return _unmatched_saved


def reset_unmatched_flag() -> None:
    """Reset the unmatched saved flag."""
    global _unmatched_saved
    _unmatched_saved = False


def _normalize(img: np.ndarray, size: tuple[int, int] = (40, 40)) -> np.ndarray:
    """Normalize image for comparison."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _load_library(library_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load all reference images from library directory."""
    if library_dir in _library_cache:
        return _library_cache[library_dir]

    library: dict[str, list[np.ndarray]] = {}
    library_dir.mkdir(parents=True, exist_ok=True)

    for image_path in library_dir.glob("*.png"):
        # Parse filename: NAME_N.png -> NAME
        name = image_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        value = parts[0]

        img = cv2.imread(str(image_path))
        if img is not None:
            if value not in library:
                library[value] = []
            library[value].append(_normalize(img))

    _library_cache[library_dir] = library
    return library


def _save_unmatched(image: np.ndarray, library_dir: Path) -> None:
    """Save unmatched image for manual review."""
    global _unmatched_saved
    _unmatched_saved = True

    unmatched_dir = library_dir / "unmatched"
    unmatched_dir.mkdir(parents=True, exist_ok=True)

    # Check for duplicates
    normalized = _normalize(image)
    for existing in unmatched_dir.glob("*.png"):
        existing_img = cv2.imread(str(existing))
        if existing_img is not None:
            if float(np.mean(np.abs(normalized - _normalize(existing_img)))) < 0.08:
                return  # Already saved

    next_index = len(list(unmatched_dir.glob("*.png")))
    cv2.imwrite(str(unmatched_dir / f"unmatched_{next_index}.png"), image)


def match_image(
    image: np.ndarray,
    library_dir: Path,
    threshold: float = 0.07,
) -> tuple[Optional[str], float]:
    """
    Match an image against a library of references.

    Args:
        image: BGR image to match
        library_dir: Directory containing reference images (NAME_N.png format)
        threshold: Maximum mean absolute difference for a match

    Returns:
        Tuple of (matched value or None, score). Score is mean absolute difference
        (lower is better, threshold is 0.07).
    """
    if image is None or image.size == 0:
        return None, 1.0

    library = _load_library(library_dir)
    if not library:
        _save_unmatched(image, library_dir)
        return None, 1.0

    normalized = _normalize(image)
    best_match: Optional[str] = None
    best_score = float("inf")

    for value, ref_images in library.items():
        for ref_img in ref_images:
            score = float(np.mean(np.abs(normalized - ref_img)))
            if score < best_score:
                best_score = score
                best_match = value

    if best_match is not None and best_score < threshold:
        return best_match, best_score

    _save_unmatched(image, library_dir)
    return None, best_score


def clear_cache() -> None:
    """Clear the library cache."""
    _library_cache.clear()
