"""
Fast OCR module using PaddleOCR for text recognition.

PaddleOCR provides faster and more accurate OCR than Tesseract,
especially for small text regions like chip amounts.
"""

from typing import Optional
import threading
import logging

import cv2
import numpy as np

# Suppress PaddleOCR's verbose logging
logging.getLogger("ppocr").setLevel(logging.WARNING)


class FastOCR:
    """
    Fast OCR using PaddleOCR with reusable model instance.

    Thread-safe singleton that maintains a PaddleOCR instance
    for fast repeated OCR calls without model reload overhead.
    """

    _instance: Optional["FastOCR"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FastOCR":
        """Singleton pattern for shared instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the OCR model (only once due to singleton)."""
        if self._initialized:
            return

        from paddleocr import PaddleOCR

        # Initialize PaddleOCR with optimized settings
        # use_angle_cls=False for speed (our images are upright)
        # show_log=False to reduce noise
        self._ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
            use_gpu=False,  # CPU is usually faster for small images
        )

        self._initialized = True

    def ocr_digits(self, image: np.ndarray) -> str:
        """
        OCR optimized for digit recognition (chip amounts, pot).

        Args:
            image: BGR numpy array.

        Returns:
            Recognized text string containing digits.
        """
        return self._ocr_image(image)

    def ocr_card_rank(self, image: np.ndarray) -> str:
        """
        OCR optimized for card rank recognition.

        Args:
            image: BGR numpy array of card rank area.

        Returns:
            Recognized text string.
        """
        return self._ocr_image(image)

    def ocr_general(self, image: np.ndarray) -> str:
        """
        General OCR for any text.

        Args:
            image: BGR numpy array.

        Returns:
            Recognized text string.
        """
        return self._ocr_image(image)

    def _ocr_image(self, image: np.ndarray) -> str:
        """Run OCR on an image and return concatenated text."""
        if image is None or image.size == 0:
            return ""

        with self._lock:
            # PaddleOCR expects BGR or RGB, handles both
            result = self._ocr.ocr(image, cls=False)

        if not result or not result[0]:
            return ""

        # Extract text from all detected regions
        texts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text, confidence = line[1]
                texts.append(text)

        return " ".join(texts)

    def close(self):
        """Release resources."""
        with self._lock:
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


def ocr_card_rank(image: np.ndarray) -> str:
    """OCR for card ranks."""
    return get_fast_ocr().ocr_card_rank(image)


def ocr_general(image: np.ndarray) -> str:
    """General OCR."""
    return get_fast_ocr().ocr_general(image)
