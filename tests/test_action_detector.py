"""Tests for player action detection module."""

import numpy as np
import pytest

from lieutenant_of_poker.action_detector import (
    PlayerAction,
    DetectedAction,
    ActionDetector,
    detect_player_action,
)


class TestPlayerAction:
    """Tests for PlayerAction enum."""

    def test_all_actions_defined(self):
        """Test all expected actions are defined."""
        assert PlayerAction.FOLD
        assert PlayerAction.CHECK
        assert PlayerAction.CALL
        assert PlayerAction.RAISE
        assert PlayerAction.BET
        assert PlayerAction.ALL_IN
        assert PlayerAction.UNKNOWN


class TestDetectedAction:
    """Tests for DetectedAction dataclass."""

    def test_str_without_amount(self):
        """Test string representation without amount."""
        action = DetectedAction(action=PlayerAction.CHECK)
        assert str(action) == "CHECK"

        action = DetectedAction(action=PlayerAction.FOLD)
        assert str(action) == "FOLD"

    def test_str_with_amount(self):
        """Test string representation with amount."""
        action = DetectedAction(action=PlayerAction.CALL, amount=100)
        assert str(action) == "CALL 100"

        action = DetectedAction(action=PlayerAction.RAISE, amount=500)
        assert str(action) == "RAISE 500"


class TestActionDetector:
    """Tests for ActionDetector class."""

    def test_initialization(self):
        """Test detector initializes without error."""
        detector = ActionDetector()
        assert detector is not None

    def test_detect_action_label_none(self):
        """Test detection returns None for invalid input."""
        detector = ActionDetector()
        assert detector.detect_action_label(None) is None
        assert detector.detect_action_label(np.array([])) is None

    def test_detect_action_buttons_empty(self):
        """Test button detection returns empty list for invalid input."""
        detector = ActionDetector()
        assert detector.detect_action_buttons(None) == []
        assert detector.detect_action_buttons(np.array([])) == []

    def test_parse_action_fold(self):
        """Test parsing FOLD action."""
        detector = ActionDetector()
        result = detector._parse_action("FOLD")
        assert result is not None
        assert result.action == PlayerAction.FOLD

    def test_parse_action_check(self):
        """Test parsing CHECK action."""
        detector = ActionDetector()
        result = detector._parse_action("CHECK")
        assert result is not None
        assert result.action == PlayerAction.CHECK

    def test_parse_action_call_with_amount(self):
        """Test parsing CALL with amount."""
        detector = ActionDetector()
        result = detector._parse_action("CALL 100")
        assert result is not None
        assert result.action == PlayerAction.CALL
        assert result.amount == 100

    def test_parse_action_raise_with_amount(self):
        """Test parsing RAISE with amount."""
        detector = ActionDetector()
        result = detector._parse_action("RAISE TO 500")
        assert result is not None
        assert result.action == PlayerAction.RAISE
        assert result.amount == 500

    def test_parse_action_all_in(self):
        """Test parsing ALL-IN action."""
        detector = ActionDetector()

        for text in ["ALL-IN", "ALL IN", "ALLIN"]:
            result = detector._parse_action(text)
            assert result is not None
            assert result.action == PlayerAction.ALL_IN

    def test_parse_action_empty(self):
        """Test parsing empty text returns None."""
        detector = ActionDetector()
        assert detector._parse_action("") is None
        assert detector._parse_action(None) is None

    def test_extract_amount(self):
        """Test amount extraction from text."""
        detector = ActionDetector()

        assert detector._extract_amount("CALL 100") == 100
        assert detector._extract_amount("RAISE TO 1000") == 1000
        assert detector._extract_amount("BET 50") == 50
        assert detector._extract_amount("CHECK") is None


class TestDetectPlayerAction:
    """Tests for convenience function."""

    def test_returns_optional_detected_action(self):
        """Test function returns DetectedAction or None."""
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = detect_player_action(img)
        assert result is None or isinstance(result, DetectedAction)
