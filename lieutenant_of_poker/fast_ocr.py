"""
Fast OCR module for digit recognition.

Uses two approaches:
- Tesseract for pot detection (works great on larger text)
- Matched filter convolution for player chips (better for smaller text)
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tesserocr
from scipy.signal import convolve2d, find_peaks

# Tesseract API for pot OCR
_tess_lock = threading.Lock()
_tess_api = None

# Load digit matched filters (10 templates for digits 0-9)
_FILTER_PATH = Path(__file__).parent / "digit_filters.npy"
_digit_filters: np.ndarray | None = None

# OCR debug image saving
_ocr_debug_dir: Path | None = None
_ocr_debug_context: dict[str, str] = {}  # source, timestamp


def _get_tess_api() -> tesserocr.PyTessBaseAPI:
    """Get or create the shared tesseract API instance."""
    global _tess_api
    if _tess_api is None:
        _tess_api = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE)
        _tess_api.SetVariable("tessedit_char_whitelist", "0123456789,.")
    return _tess_api


def _preprocess_for_tesseract(image: np.ndarray) -> np.ndarray:
    """Prepare image for tesseract OCR."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert - tesseract works better with dark text on light background
    image = 255 - image
    # Boost contrast
    image = np.where(image > 200, 255, image).astype(np.uint8)
    return image


# Public alias for diagnostic module
preprocess_for_ocr = _preprocess_for_tesseract


def _get_filters() -> np.ndarray:
    """Load and cache the digit matched filters."""
    global _digit_filters
    if _digit_filters is None:
        _digit_filters = np.load(str(_FILTER_PATH))
    return _digit_filters


def _load_and_normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to normalized grayscale array."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    arr = gray.astype(np.float32)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    return arr


def _do_conv(image: np.ndarray) -> np.ndarray:
    """Apply digit filters via convolution."""
    im = _load_and_normalize_grayscale(image)
    im = np.pad(im, ((0, 0), (2, 0)))

    filters = _get_filters()
    ims = []
    for i in range(10):
        c = convolve2d(im, filters[::-1, ::-1, i], mode='valid')
        c = np.max(c, 0)
        ims.append(c)
    ims = np.stack(ims, -1)
    return ims


def _get_numbers_matched_filter(image: np.ndarray) -> np.ndarray:
    """Detect digits in image using matched filter convolution."""
    c = _do_conv(image)
    peaks, _ = find_peaks(np.max(c, 1), height=0.99, distance=10)
    args = np.argmax(c, 1)
    return args[peaks]


def _get_numbers_tesseract(image: np.ndarray) -> str:
    """Detect digits using tesseract OCR."""
    with _tess_lock:
        _get_tess_api().SetImage(Image.fromarray(_preprocess_for_tesseract(image)))
        return _get_tess_api().GetUTF8Text().strip()


def enable_ocr_debug(directory: Path | str) -> None:
    """
    Enable saving of OCR input images for debugging.

    Args:
        directory: Directory to save images to.
    """
    global _ocr_debug_dir, _ocr_debug_context
    _ocr_debug_dir = Path(directory)
    _ocr_debug_dir.mkdir(parents=True, exist_ok=True)
    _ocr_debug_context = {}


def disable_ocr_debug() -> None:
    """Disable saving of OCR input images."""
    global _ocr_debug_dir
    _ocr_debug_dir = None


def set_ocr_debug_context(source: str, timestamp_ms: float) -> None:
    """Set context for OCR debug filenames (call before processing each frame)."""
    global _ocr_debug_context
    _ocr_debug_context = {
        "source": Path(source).stem,  # filename without extension
        "timestamp": f"{int(timestamp_ms)}ms",
    }


def ocr_digits(image: np.ndarray, category: str = "other") -> str:
    """
    OCR optimized for digit recognition (chip amounts, pot).

    Args:
        image: BGR or grayscale numpy array.
        category: Category - "pot" uses tesseract, others use matched filter.

    Returns:
        Recognized text string of digits.
    """
    import secrets

    # Use tesseract for pot (works great), matched filter for players
    if category == "pot":
        result = _get_numbers_tesseract(image)
    else:
        digits = _get_numbers_matched_filter(image)
        result = "".join(str(d) for d in digits)

    # Save image with OCR result in filename if debug enabled
    if _ocr_debug_dir is not None:
        # Create category subdirectory
        category_dir = _ocr_debug_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Build filename: source_timestamp_hash_result.png
        source = _ocr_debug_context.get("source", "unknown")
        timestamp = _ocr_debug_context.get("timestamp", "0ms")
        random_hash = secrets.token_hex(3)  # 6 char hex

        # Sanitize result for filename (remove commas/periods from tesseract output)
        safe_result = result.replace(",", "").replace(".", "").replace(" ", "")
        if not safe_result:
            safe_result = "EMPTY"

        filename = f"{source}_{timestamp}_{random_hash}_{safe_result}.png"
        cv2.imwrite(str(category_dir / filename), image)

    return result
