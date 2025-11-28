"""Tests for command line interface."""

import json
import subprocess
import sys
import pytest


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self):
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Lieutenant of Poker" in result.stdout
        assert "extract-frames" in result.stdout
        assert "analyze" in result.stdout
        assert "export" in result.stdout
        assert "info" in result.stdout

    def test_version(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_command(self):
        """Test running without command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "Lieutenant of Poker" in result.stdout

    def test_extract_frames_help(self):
        """Test extract-frames --help."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "extract-frames", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--interval" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--format" in result.stdout

    def test_analyze_help(self):
        """Test analyze --help."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "analyze", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--interval" in result.stdout
        assert "--output" in result.stdout
        assert "--verbose" in result.stdout

    def test_export_help(self):
        """Test export --help."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "export", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--format" in result.stdout
        assert "pokerstars" in result.stdout

    def test_info_help(self):
        """Test info --help."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "info", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_info_nonexistent_file(self):
        """Test info with nonexistent file."""
        result = subprocess.run(
            [sys.executable, "-m", "lieutenant_of_poker.cli", "info", "/nonexistent/video.mp4"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "Error" in result.stderr


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_game_state_to_dict(self):
        """Test game state serialization."""
        from lieutenant_of_poker.cli import game_state_to_dict
        from lieutenant_of_poker.game_state import GameState, Street
        from lieutenant_of_poker.card_detector import Card, Suit, Rank

        state = GameState(
            frame_number=100,
            timestamp_ms=5000.0,
            street=Street.FLOP,
            pot=1000,
            community_cards=[Card(Rank.ACE, Suit.HEARTS)],
        )

        result = game_state_to_dict(state)

        assert result["frame_number"] == 100
        assert result["timestamp_ms"] == 5000.0
        assert result["street"] == "FLOP"
        assert result["pot"] == 1000
        assert "A♥" in result["community_cards"]

    def test_hand_to_dict(self):
        """Test hand history serialization."""
        from lieutenant_of_poker.cli import hand_to_dict
        from lieutenant_of_poker.hand_history import HandHistory
        from lieutenant_of_poker.card_detector import Card, Suit, Rank

        hand = HandHistory(
            hand_id="123",
            pot=500,
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
        )

        result = hand_to_dict(hand)

        assert result["hand_id"] == "123"
        assert result["pot"] == 500
        assert "As" in result["hero_cards"]
        assert "Ks" in result["hero_cards"]
