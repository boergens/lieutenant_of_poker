"""
Image preprocessing for Governor of Poker analysis.

Game-specific preprocessing that happens before OCR or other analysis.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Load reference green color from card library
_GREEN_REFERENCE_PATH = Path(__file__).parent / "card_library" / "green.png"
_GREEN_BGR: Optional[np.ndarray] = None


def _get_green_reference() -> np.ndarray:
    """Load and cache the green reference color from card library."""
    global _GREEN_BGR
    if _GREEN_BGR is None:
        green_img = cv2.imread(str(_GREEN_REFERENCE_PATH))
        if green_img is not None:
            _GREEN_BGR = np.mean(green_img, axis=(0, 1)).astype(np.float32)
        else:
            # Fallback to approximate green if file not found
            _GREEN_BGR = np.array([24.0, 122.0, 105.0], dtype=np.float32)
    return _GREEN_BGR


def _map_green_to_dark(image: np.ndarray) -> np.ndarray:
    """
    Map pixels close to reference green to dark (10,10,10).

    Args:
        image: BGR image array.

    Returns:
        Image with green pixels mapped to dark.
    """
    green_ref = _get_green_reference()
    green_tolerance = 40

    # Calculate distance from each pixel to reference green
    distances = np.linalg.norm(image.astype(np.float32) - green_ref, axis=-1)

    # Create mask of green pixels
    green_mask = distances < green_tolerance

    # Map green pixels to dark
    result = image.copy()
    result[green_mask] = [10, 10, 10]

    return result


def _is_strip_significant(pixels: np.ndarray) -> bool:
    """
    Check if a strip of pixels (row or column) contains significant dark content.

    Args:
        pixels: Pixel strip as Nx3 BGR array.

    Returns:
        True if strip has at least 75% dark pixels, False otherwise.
    """
    darkness_threshold = 80
    pixel_brightness = np.mean(pixels, axis=-1)
    dark_pixel_count = np.sum(pixel_brightness < darkness_threshold)
    return dark_pixel_count > len(pixels) * 3 // 4


def trim_to_content(image: np.ndarray) -> np.ndarray:
    """
    Trim empty edges from image until hitting content.

    First maps green background to dark, then trims non-dark edges.

    Args:
        image: BGR image array.

    Returns:
        Trimmed image with empty edges removed.
    """
    if len(image.shape) != 3:
        return image

    # Map green to dark first
    image = _map_green_to_dark(image)

    # Trim from left
    left = 0
    while left < image.shape[1] - 1:
        if _is_strip_significant(image[:, left, :]):
            break
        left += 1
    if left > 0:
        image = image[:, left:]

    # Trim from top
    top = 0
    while top < image.shape[0] - 1:
        if _is_strip_significant(image[top, :, :]):
            break
        top += 1
    if top > 0:
        image = image[top:, :]

    # Trim from bottom
    bottom = image.shape[0] - 1
    while bottom > 0:
        if _is_strip_significant(image[bottom, :, :]):
            break
        bottom -= 1
    if bottom < image.shape[0] - 1:
        image = image[: bottom + 1, :]

    return image
