"""Tests for game state aggregator module."""

import numpy as np
import pytest

from lieutenant_of_poker.game_state import (
    Street,
    PlayerState,
    GameState,
    GameStateExtractor,
    extract_game_state,
)
from lieutenant_of_poker.table_regions import PlayerPosition
from lieutenant_of_poker.card_detector import Card, Suit, Rank
from lieutenant_of_poker.action_detector import DetectedAction, PlayerAction


class TestStreet:
    """Tests for Street enum."""

    def test_all_streets_defined(self):
        """Test all streets are defined."""
        assert Street.PREFLOP
        assert Street.FLOP
        assert Street.TURN
        assert Street.RIVER
        assert Street.SHOWDOWN
        assert Street.UNKNOWN


class TestPlayerState:
    """Tests for PlayerState dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        state = PlayerState(position=PlayerPosition.BOTTOM)
        assert state.position == PlayerPosition.BOTTOM
        assert state.name is None
        assert state.chips is None
        assert state.cards == []
        assert state.last_action is None
        assert state.is_active is True
        assert state.is_dealer is False


class TestGameState:
    """Tests for GameState dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        state = GameState()
        assert state.community_cards == []
        assert state.hero_cards == []
        assert state.pot is None
        assert state.hero_chips is None
        assert state.players == {}
        assert state.street == Street.UNKNOWN

    def test_determine_street_preflop(self):
        """Test street determination with no community cards."""
        state = GameState(community_cards=[])
        assert state.determine_street() == Street.PREFLOP

    def test_determine_street_flop(self):
        """Test street determination with 3 community cards."""
        state = GameState(community_cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
        ])
        assert state.determine_street() == Street.FLOP

    def test_determine_street_turn(self):
        """Test street determination with 4 community cards."""
        state = GameState(community_cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
        ])
        assert state.determine_street() == Street.TURN

    def test_determine_street_river(self):
        """Test street determination with 5 community cards."""
        state = GameState(community_cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
        ])
        assert state.determine_street() == Street.RIVER

    def test_str_representation(self):
        """Test string representation."""
        state = GameState(
            street=Street.FLOP,
            pot=1000,
            community_cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.KING, Suit.HEARTS),
                Card(Rank.QUEEN, Suit.HEARTS),
            ],
            hero_cards=[
                Card(Rank.ACE, Suit.SPADES),
                Card(Rank.ACE, Suit.DIAMONDS),
            ],
            hero_chips=5000
        )
        s = str(state)
        assert "FLOP" in s
        assert "1,000" in s
        assert "A♥" in s
        assert "A♠" in s
        assert "5,000" in s

    def test_num_community_cards(self):
        """Test community card count property."""
        state = GameState()
        assert state.num_community_cards == 0

        state.community_cards = [Card(Rank.ACE, Suit.HEARTS)]
        assert state.num_community_cards == 1


class TestGameStateExtractor:
    """Tests for GameStateExtractor class."""

    def test_initialization(self):
        """Test extractor initializes without error."""
        extractor = GameStateExtractor()
        assert extractor is not None
        assert extractor.card_detector is not None
        assert extractor.chip_ocr is not None
        assert extractor.action_detector is not None

    def test_extract_returns_game_state(self):
        """Test extract returns a GameState object."""
        extractor = GameStateExtractor()
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)
        state = extractor.extract(frame)
        assert isinstance(state, GameState)

    def test_extract_with_metadata(self):
        """Test extract preserves frame metadata."""
        extractor = GameStateExtractor()
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)
        state = extractor.extract(frame, frame_number=100, timestamp_ms=5000.0)
        assert state.frame_number == 100
        assert state.timestamp_ms == 5000.0


class TestExtractGameState:
    """Tests for convenience function."""

    def test_returns_game_state(self):
        """Test function returns GameState."""
        frame = np.zeros((1117, 1728, 3), dtype=np.uint8)
        state = extract_game_state(frame)
        assert isinstance(state, GameState)
