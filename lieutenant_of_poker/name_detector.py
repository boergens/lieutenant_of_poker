"""
One-time player name detection for Governor of Poker.

Player names are static during a game session, so this module provides
a function to detect names once from a single frame and cache them.
"""

import threading
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
import tesserocr

from .table_regions import (
    TableRegionDetector,
    PlayerPosition,
    Region,
    BASE_WIDTH,
    BASE_HEIGHT,
)


# Name regions positioned above chip regions (at BASE resolution)
# These are separate from the per-frame chip detection regions
_NAME_REGIONS: Dict[PlayerPosition, Region] = {
    PlayerPosition.SEAT_1: Region(x=287, y=618, width=150, height=28),
    PlayerPosition.SEAT_2: Region(x=439, y=217, width=144, height=28),
    PlayerPosition.SEAT_3: Region(x=1179, y=223, width=147, height=28),
    PlayerPosition.SEAT_4: Region(x=1341, y=617, width=143, height=28),
    PlayerPosition.HERO: Region(x=974, y=817, width=128, height=28),
}

# Tesseract API for name OCR (separate from digit OCR)
_tess_lock = threading.Lock()
_tess_api: Optional[tesserocr.PyTessBaseAPI] = None


def _get_tess_api() -> tesserocr.PyTessBaseAPI:
    """Get or create the shared tesseract API for text recognition."""
    global _tess_api
    if _tess_api is None:
        _tess_api = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE)
        # Allow alphanumeric characters for player names
        _tess_api.SetVariable("tessedit_char_whitelist",
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
    return _tess_api


def _preprocess_for_name_ocr(image: np.ndarray) -> np.ndarray:
    """Prepare image for name OCR."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert - tesseract works better with dark text on light background
    image = 255 - image
    # Boost contrast
    image = np.where(image > 180, 255, image).astype(np.uint8)
    return image


def _ocr_name(image: np.ndarray) -> Optional[str]:
    """Extract player name from a region image."""
    if image is None or image.size == 0:
        return None

    processed = _preprocess_for_name_ocr(image)

    with _tess_lock:
        api = _get_tess_api()
        api.SetImage(Image.fromarray(processed))
        text = api.GetUTF8Text().strip()

    # Filter out very short results (likely noise)
    if len(text) < 2:
        return None

    return text


def detect_player_names(
    frame: np.ndarray,
    scale_frame: bool = True,
) -> Dict[PlayerPosition, Optional[str]]:
    """
    Detect player names from a single frame.

    This should be called once at the start of analysis, not on every frame.

    Args:
        frame: BGR image frame from the game.
        scale_frame: If True, scale frame to base resolution first.

    Returns:
        Dictionary mapping PlayerPosition to detected name (or None if not found).
    """
    # Scale frame if needed (same logic as GameStateExtractor)
    if scale_frame:
        h, w = frame.shape[:2]
        if w > BASE_WIDTH or h > BASE_HEIGHT:
            scale = min(BASE_WIDTH / w, BASE_HEIGHT / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Get scaling factors for regions
    h, w = frame.shape[:2]
    scale_x = w / BASE_WIDTH
    scale_y = h / BASE_HEIGHT

    names: Dict[PlayerPosition, Optional[str]] = {}

    for position, base_region in _NAME_REGIONS.items():
        # Scale region to match frame
        scaled_region = base_region.scale(scale_x, scale_y)

        # Extract and OCR
        name_img = scaled_region.extract(frame)
        name = _ocr_name(name_img)
        names[position] = name

    return names


def extract_name_region(
    frame: np.ndarray,
    position: PlayerPosition,
    scale_frame: bool = True,
) -> np.ndarray:
    """
    Extract the name region image for a specific player position.

    Useful for debugging/visualization.

    Args:
        frame: BGR image frame.
        position: Player position to extract.
        scale_frame: If True, scale frame to base resolution first.

    Returns:
        The extracted name region image.
    """
    if scale_frame:
        h, w = frame.shape[:2]
        if w > BASE_WIDTH or h > BASE_HEIGHT:
            scale = min(BASE_WIDTH / w, BASE_HEIGHT / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    scale_x = w / BASE_WIDTH
    scale_y = h / BASE_HEIGHT

    base_region = _NAME_REGIONS[position]
    scaled_region = base_region.scale(scale_x, scale_y)

    return scaled_region.extract(frame)
