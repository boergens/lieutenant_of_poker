"""
Fast OCR module for digit recognition.

Uses two approaches:
- PaddleOCR for pot detection (works great on larger text)
- Matched filter convolution for player chips (better for smaller text)
"""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tesserocr
from scipy.signal import convolve2d, find_peaks

# Suppress PaddleOCR's verbose logging
logging.getLogger("ppocr").setLevel(logging.WARNING)

# PaddleOCR instance (lazy loaded)
_paddle_ocr = None
_paddle_lock = threading.Lock()

# Tesseract API for name OCR (keep for names which PaddleOCR isn't great at)
_tess_lock = threading.Lock()
_tess_name_api = None
_DISALLOWED_NAME_CHARS = __import__('re').compile(r'[^A-Za-z0-9_]')

# Load digit matched filters (10 templates for digits 0-9)
_FILTER_PATH = Path(__file__).parent / "digit_filters.npy"
_digit_filters: np.ndarray | None = None

# OCR debug image saving
_ocr_debug_dir: Path | None = None
_ocr_debug_context: dict[str, str] = {}  # source, timestamp


def _get_paddle_ocr():
    """Get or create the shared PaddleOCR instance."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _paddle_ocr


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Prepare image for OCR - just inversion."""
    # Invert - OCR works better with dark text on light background
    return 255 - image


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


def _get_numbers_paddle(image: np.ndarray) -> str:
    """Detect digits using PaddleOCR."""
    if image is None or image.size == 0:
        return ""

    image = preprocess_for_ocr(image)

    with _paddle_lock:
        ocr = _get_paddle_ocr()
        result = ocr.predict(image)

    if not result:
        return ""

    # Extract text from result - new API returns list of dicts
    texts = []
    for item in result:
        if isinstance(item, dict) and "rec_texts" in item:
            texts.extend(item["rec_texts"])

    return " ".join(texts)


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
        category: Category - "pot" uses PaddleOCR, others use matched filter.

    Returns:
        Recognized text string of digits.
    """
    import secrets

    # Use PaddleOCR for pot and money, matched filter for players
    if category in ("pot", "money"):
        result = _get_numbers_paddle(image)
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


_tess_name_api_word = None


def _get_tess_name_api() -> tesserocr.PyTessBaseAPI:
    """Get or create tesseract API for name OCR (SINGLE_LINE mode)."""
    global _tess_name_api
    if _tess_name_api is None:
        _tess_name_api = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE)
        _tess_name_api.SetVariable(
            "tessedit_char_whitelist",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        )
    return _tess_name_api


def _get_tess_name_api_word() -> tesserocr.PyTessBaseAPI:
    """Get or create tesseract API for name OCR (SINGLE_WORD fallback mode)."""
    global _tess_name_api_word
    if _tess_name_api_word is None:
        _tess_name_api_word = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_WORD)
        _tess_name_api_word.SetVariable(
            "tessedit_char_whitelist",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        )
    return _tess_name_api_word


def ocr_name(image: np.ndarray) -> str | None:
    """
    OCR for player names.

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Recognized name string, or None if too short.
    """
    processed = preprocess_for_ocr(image)

    with _tess_lock:
        api = _get_tess_name_api()
        api.SetImage(Image.fromarray(processed))
        text = api.GetUTF8Text().strip()

        # Fallback to SINGLE_WORD mode if SINGLE_LINE returns nothing
        if not text:
            api_word = _get_tess_name_api_word()
            api_word.SetImage(Image.fromarray(processed))
            text = api_word.GetUTF8Text().strip()

    # Sanitize: replace spaces with underscores, remove invalid chars
    text = text.replace(' ', '_')
    text = _DISALLOWED_NAME_CHARS.sub('', text)

    # Filter out very short results (likely noise)
    if len(text) < 2:
        return None

    return text


# Name region constants (match TableInfo in first_frame.py)
_NAME_WIDTH = 140
_NAME_HEIGHT = 30


def get_name_region(image: np.ndarray, pos: tuple[int, int]) -> np.ndarray | None:
    """
    Extract the name region at a seat position.

    Args:
        image: The frame image (BGR).
        pos: (x, y) seat position coordinates.

    Returns:
        Name region image or None if out of bounds.
    """
    pos_x, pos_y = pos
    x = pos_x
    y = pos_y - _NAME_HEIGHT

    # Bounds check
    if y < 0 or x < 0:
        return None
    if y + _NAME_HEIGHT > image.shape[0] or x + _NAME_WIDTH > image.shape[1]:
        return None

    return image[y:y + _NAME_HEIGHT, x:x + _NAME_WIDTH]


def ocr_name_at_position(image: np.ndarray, pos: tuple[int, int]) -> str | None:
    """
    Extract player name at a seat position.

    The name region is located above the seat position coordinates.

    Args:
        image: The frame image (BGR).
        pos: (x, y) seat position coordinates.

    Returns:
        Detected name string or None.
    """
    region = get_name_region(image, pos)
    if region is None:
        return None
    return ocr_name(region)
