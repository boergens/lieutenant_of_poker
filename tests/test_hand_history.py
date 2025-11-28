"""Tests for hand history export module."""

import pytest

from lieutenant_of_poker.hand_history import (
    HandAction,
    HandHistory,
    HandHistoryExporter,
    export_hand_to_pokerstars,
)
from lieutenant_of_poker.card_detector import Card, Suit, Rank
from lieutenant_of_poker.action_detector import PlayerAction
from lieutenant_of_poker.game_state import GameState


class TestHandAction:
    """Tests for HandAction dataclass."""

    def test_str_fold(self):
        """Test fold action string."""
        action = HandAction("Player1", PlayerAction.FOLD)
        assert str(action) == "Player1: folds"

    def test_str_check(self):
        """Test check action string."""
        action = HandAction("Player1", PlayerAction.CHECK)
        assert str(action) == "Player1: checks"

    def test_str_call_with_amount(self):
        """Test call action with amount."""
        action = HandAction("Player1", PlayerAction.CALL, 100)
        assert str(action) == "Player1: calls $100"

    def test_str_raise_with_amount(self):
        """Test raise action with amount."""
        action = HandAction("Player1", PlayerAction.RAISE, 200)
        assert str(action) == "Player1: raises to $200"

    def test_str_bet_with_amount(self):
        """Test bet action with amount."""
        action = HandAction("Player1", PlayerAction.BET, 50)
        assert str(action) == "Player1: bets $50"

    def test_str_all_in(self):
        """Test all-in action."""
        action = HandAction("Player1", PlayerAction.ALL_IN, 1000)
        assert "all-in" in str(action)
        assert "1000" in str(action)


class TestHandHistory:
    """Tests for HandHistory dataclass."""

    def test_default_values(self):
        """Test default values."""
        hand = HandHistory(hand_id="1")
        assert hand.hand_id == "1"
        assert hand.small_blind == 10
        assert hand.big_blind == 20
        assert hand.players == []
        assert hand.hero_cards == []

    def test_with_cards(self):
        """Test hand with cards."""
        hand = HandHistory(
            hand_id="1",
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)],
            flop_cards=[
                Card(Rank.QUEEN, Suit.DIAMONDS),
                Card(Rank.JACK, Suit.CLUBS),
                Card(Rank.TEN, Suit.SPADES),
            ],
        )
        assert len(hand.hero_cards) == 2
        assert len(hand.flop_cards) == 3


class TestHandHistoryExporter:
    """Tests for HandHistoryExporter class."""

    def test_initialization(self):
        """Test exporter initializes."""
        exporter = HandHistoryExporter()
        assert exporter is not None

    def test_export_basic_hand(self):
        """Test exporting a basic hand."""
        exporter = HandHistoryExporter()
        hand = HandHistory(
            hand_id="123",
            players=[("Hero", 1000, 0), ("Villain", 1000, 1)],
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
        )
        result = exporter.export_pokerstars_format(hand)

        assert "PokerStars Hand #123" in result
        assert "Hold'em No Limit" in result
        assert "Hero" in result
        assert "Villain" in result
        assert "As Ks" in result

    def test_export_with_community_cards(self):
        """Test export includes community cards."""
        exporter = HandHistoryExporter()
        hand = HandHistory(
            hand_id="456",
            players=[("Hero", 1000, 0)],
            flop_cards=[
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.JACK, Suit.HEARTS),
                Card(Rank.TEN, Suit.HEARTS),
            ],
            turn_card=Card(Rank.TWO, Suit.CLUBS),
            river_card=Card(Rank.THREE, Suit.DIAMONDS),
        )
        result = exporter.export_pokerstars_format(hand)

        assert "*** FLOP ***" in result
        assert "*** TURN ***" in result
        assert "*** RIVER ***" in result
        assert "Qh Jh 10h" in result

    def test_export_with_actions(self):
        """Test export includes actions."""
        exporter = HandHistoryExporter()
        hand = HandHistory(
            hand_id="789",
            players=[("Hero", 1000, 0), ("Villain", 1000, 1)],
            preflop_actions=[
                HandAction("Hero", PlayerAction.RAISE, 60),
                HandAction("Villain", PlayerAction.CALL, 60),
            ],
        )
        result = exporter.export_pokerstars_format(hand)

        assert "raises to $60" in result
        assert "calls $60" in result

    def test_format_card(self):
        """Test card formatting."""
        exporter = HandHistoryExporter()
        card = Card(Rank.ACE, Suit.HEARTS)
        assert exporter._format_card(card) == "Ah"

        card = Card(Rank.TEN, Suit.SPADES)
        assert exporter._format_card(card) == "10s"

    def test_create_hand_from_empty_states(self):
        """Test creating hand from empty states returns None."""
        exporter = HandHistoryExporter()
        result = exporter.create_hand_from_states([])
        assert result is None

    def test_create_hand_from_states(self):
        """Test creating hand from game states."""
        exporter = HandHistoryExporter()
        states = [
            GameState(
                hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
                community_cards=[],
                pot=30,
            ),
            GameState(
                hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
                community_cards=[
                    Card(Rank.QUEEN, Suit.HEARTS),
                    Card(Rank.JACK, Suit.HEARTS),
                    Card(Rank.TEN, Suit.HEARTS),
                ],
                pot=100,
            ),
        ]
        hand = exporter.create_hand_from_states(states)

        assert hand is not None
        assert len(hand.hero_cards) == 2
        assert len(hand.flop_cards) == 3
        assert hand.pot == 100


class TestExportHandToPokerstars:
    """Tests for convenience function."""

    def test_returns_string(self):
        """Test function returns a string."""
        hand = HandHistory(hand_id="1")
        result = export_hand_to_pokerstars(hand)
        assert isinstance(result, str)
        assert "PokerStars" in result
