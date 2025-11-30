"""
Fast OCR module using tesserocr for low-latency text recognition.

Uses tesserocr (C API bindings) instead of pytesseract (subprocess)
for faster OCR calls by keeping the tesseract engine loaded.
"""

from typing import Optional
import threading

import cv2
import numpy as np
from PIL import Image
import tesserocr


class FastOCR:
    """
    Fast OCR using tesserocr with reusable API instance.

    Thread-safe singleton that maintains a tesserocr API instance
    for fast repeated OCR calls without subprocess overhead.
    """

    _instance: Optional["FastOCR"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FastOCR":
        """Singleton pattern for shared API instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the OCR API (only once due to singleton)."""
        if self._initialized:
            return

        # Create API instance for digit recognition
        self._api_digits = tesserocr.PyTessBaseAPI(
            psm=tesserocr.PSM.SINGLE_LINE
        )
        self._api_digits.SetVariable("tessedit_char_whitelist", "0123456789,.")

        self._api_general = tesserocr.PyTessBaseAPI(
            psm=tesserocr.PSM.SINGLE_LINE
        )

        self._initialized = True

    def ocr_digits(self, image: np.ndarray) -> str:
        """
        OCR optimized for digit recognition (chip amounts, pot).

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Recognized text string.
        """
        pil_image = self._to_pil(image)
        with self._lock:
            self._api_digits.SetImage(pil_image)
            return self._api_digits.GetUTF8Text().strip()

    def ocr_general(self, image: np.ndarray) -> str:
        """
        General OCR for any text.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Recognized text string.
        """
        pil_image = self._to_pil(image)
        with self._lock:
            self._api_general.SetImage(pil_image)
            return self._api_general.GetUTF8Text().strip()

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image using shared preprocessing."""
        preprocessed = preprocess_for_ocr(image)
        return Image.fromarray(preprocessed)

    def close(self):
        """Release tesserocr resources."""
        with self._lock:
            if hasattr(self, '_api_digits'):
                self._api_digits.End()
            if hasattr(self, '_api_general'):
                self._api_general.End()
            self._initialized = False
            FastOCR._instance = None


# Module-level convenience functions
_fast_ocr: Optional[FastOCR] = None


def get_fast_ocr() -> FastOCR:
    """Get the shared FastOCR instance."""
    global _fast_ocr
    if _fast_ocr is None:
        _fast_ocr = FastOCR()
    return _fast_ocr


def ocr_digits(image: np.ndarray) -> str:
    """OCR for digits (chip amounts)."""
    return get_fast_ocr().ocr_digits(image)


def ocr_general(image: np.ndarray) -> str:
    """General OCR."""
    return get_fast_ocr().ocr_general(image)


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Return the preprocessed image that would be sent to tesseract.

    Useful for diagnostics to see exactly what OCR receives.

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Preprocessed image as numpy array (grayscale, inverted).
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert colors - tesseract works better with dark text on light background
    return 255 - image
