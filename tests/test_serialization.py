"""Tests for serialization module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from lieutenant_of_poker.serialization import (
    card_to_dict,
    card_from_dict,
    game_state_to_dict,
    game_state_from_dict,
    hand_history_to_dict,
    hand_history_from_dict,
    player_state_to_dict,
    player_state_from_dict,
    hand_action_to_dict,
    hand_action_from_dict,
    player_info_to_dict,
    player_info_from_dict,
    save_game_states,
    load_game_states,
    save_hand_history,
    load_hand_history,
)
from lieutenant_of_poker.game_state import GameState, PlayerState, Street
from lieutenant_of_poker.hand_history import HandHistory, HandAction, PlayerInfo
from lieutenant_of_poker.card_detector import Card, Rank, Suit
from lieutenant_of_poker.action_detector import PlayerAction, DetectedAction
from lieutenant_of_poker.table_regions import PlayerPosition


class TestCardSerialization:
    """Tests for Card serialization."""

    def test_card_roundtrip(self):
        """Card survives serialization roundtrip."""
        card = Card(Rank.ACE, Suit.SPADES)
        d = card_to_dict(card)
        restored = card_from_dict(d)

        assert restored.rank == card.rank
        assert restored.suit == card.suit

    def test_card_dict_format(self):
        """Card dict has expected format."""
        card = Card(Rank.TEN, Suit.HEARTS)
        d = card_to_dict(card)

        assert d == {"rank": "10", "suit": "hearts"}

    def test_all_ranks(self):
        """All ranks serialize correctly."""
        for rank in Rank:
            card = Card(rank, Suit.CLUBS)
            d = card_to_dict(card)
            restored = card_from_dict(d)
            assert restored.rank == rank

    def test_all_suits(self):
        """All suits serialize correctly."""
        for suit in Suit:
            card = Card(Rank.KING, suit)
            d = card_to_dict(card)
            restored = card_from_dict(d)
            assert restored.suit == suit


class TestPlayerStateSerialization:
    """Tests for PlayerState serialization."""

    def test_player_state_minimal(self):
        """PlayerState with minimal fields roundtrips."""
        state = PlayerState(position=PlayerPosition.HERO)
        d = player_state_to_dict(state)
        restored = player_state_from_dict(d)

        assert restored.position == PlayerPosition.HERO
        assert restored.chips is None
        assert restored.cards == []

    def test_player_state_full(self):
        """PlayerState with all fields roundtrips."""
        state = PlayerState(
            position=PlayerPosition.SEAT_1,
            name="Player1",
            chips=1000,
            cards=[Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)],
            last_action=DetectedAction(PlayerAction.RAISE, 100, 0.95),
            is_active=True,
            is_dealer=True,
        )
        d = player_state_to_dict(state)
        restored = player_state_from_dict(d)

        assert restored.position == PlayerPosition.SEAT_1
        assert restored.name == "Player1"
        assert restored.chips == 1000
        assert len(restored.cards) == 2
        assert restored.cards[0].rank == Rank.ACE
        assert restored.last_action.action == PlayerAction.RAISE
        assert restored.last_action.amount == 100
        assert restored.is_dealer is True


class TestGameStateSerialization:
    """Tests for GameState serialization."""

    def test_game_state_empty(self):
        """Empty GameState roundtrips."""
        state = GameState()
        d = game_state_to_dict(state)
        restored = game_state_from_dict(d)

        assert restored.community_cards == []
        assert restored.hero_cards == []
        assert restored.pot is None
        assert restored.street == Street.UNKNOWN

    def test_game_state_preflop(self):
        """Preflop GameState roundtrips."""
        state = GameState(
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
            pot=60,
            hero_chips=1940,
            street=Street.PREFLOP,
            frame_number=100,
            timestamp_ms=5000.0,
        )
        state.players[PlayerPosition.HERO] = PlayerState(
            position=PlayerPosition.HERO,
            chips=1940,
        )
        state.players[PlayerPosition.SEAT_1] = PlayerState(
            position=PlayerPosition.SEAT_1,
            chips=2000,
        )

        d = game_state_to_dict(state)
        restored = game_state_from_dict(d)

        assert len(restored.hero_cards) == 2
        assert restored.hero_cards[0].short_name == "As"
        assert restored.pot == 60
        assert restored.street == Street.PREFLOP
        assert restored.frame_number == 100
        assert len(restored.players) == 2

    def test_game_state_flop(self):
        """Flop GameState roundtrips."""
        state = GameState(
            community_cards=[
                Card(Rank.TWO, Suit.HEARTS),
                Card(Rank.SEVEN, Suit.CLUBS),
                Card(Rank.JACK, Suit.DIAMONDS),
            ],
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
            pot=200,
            street=Street.FLOP,
        )

        d = game_state_to_dict(state)
        restored = game_state_from_dict(d)

        assert len(restored.community_cards) == 3
        assert restored.street == Street.FLOP

    def test_game_state_river(self):
        """River GameState with 5 community cards roundtrips."""
        state = GameState(
            community_cards=[
                Card(Rank.TWO, Suit.HEARTS),
                Card(Rank.SEVEN, Suit.CLUBS),
                Card(Rank.JACK, Suit.DIAMONDS),
                Card(Rank.QUEEN, Suit.SPADES),
                Card(Rank.ACE, Suit.HEARTS),
            ],
            street=Street.RIVER,
        )

        d = game_state_to_dict(state)
        restored = game_state_from_dict(d)

        assert len(restored.community_cards) == 5
        assert restored.street == Street.RIVER


class TestHandActionSerialization:
    """Tests for HandAction serialization."""

    def test_hand_action_with_amount(self):
        """HandAction with amount roundtrips."""
        action = HandAction("hero", PlayerAction.RAISE, 100)
        d = hand_action_to_dict(action)
        restored = hand_action_from_dict(d)

        assert restored.player_name == "hero"
        assert restored.action == PlayerAction.RAISE
        assert restored.amount == 100

    def test_hand_action_without_amount(self):
        """HandAction without amount roundtrips."""
        action = HandAction("Player1", PlayerAction.CHECK)
        d = hand_action_to_dict(action)
        restored = hand_action_from_dict(d)

        assert restored.player_name == "Player1"
        assert restored.action == PlayerAction.CHECK
        assert restored.amount is None

    def test_all_action_types(self):
        """All action types serialize correctly."""
        for action_type in PlayerAction:
            action = HandAction("test", action_type, 50)
            d = hand_action_to_dict(action)
            restored = hand_action_from_dict(d)
            assert restored.action == action_type


class TestPlayerInfoSerialization:
    """Tests for PlayerInfo serialization."""

    def test_player_info_roundtrip(self):
        """PlayerInfo roundtrips."""
        info = PlayerInfo(
            seat=0,
            name="hero",
            chips=2000,
            position=PlayerPosition.HERO,
            is_hero=True,
        )
        d = player_info_to_dict(info)
        restored = player_info_from_dict(d)

        assert restored.seat == 0
        assert restored.name == "hero"
        assert restored.chips == 2000
        assert restored.position == PlayerPosition.HERO
        assert restored.is_hero is True


class TestHandHistorySerialization:
    """Tests for HandHistory serialization."""

    def test_hand_history_minimal(self):
        """Minimal HandHistory roundtrips."""
        hand = HandHistory(
            hand_id="12345",
            players=[
                PlayerInfo(0, "hero", 2000, PlayerPosition.HERO, True),
                PlayerInfo(1, "villain", 2000, PlayerPosition.SEAT_1, False),
            ],
        )

        d = hand_history_to_dict(hand)
        restored = hand_history_from_dict(d)

        assert restored.hand_id == "12345"
        assert len(restored.players) == 2

    def test_hand_history_full(self):
        """Full HandHistory roundtrips."""
        hand = HandHistory(
            hand_id="67890",
            table_name="Test Table",
            timestamp=datetime(2024, 1, 15, 12, 30, 0),
            small_blind=10,
            big_blind=20,
            players=[
                PlayerInfo(0, "hero", 2000, PlayerPosition.HERO, True),
                PlayerInfo(1, "villain", 1500, PlayerPosition.SEAT_1, False),
            ],
            button_seat=0,
            sb_seat=0,
            bb_seat=1,
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
            flop_cards=[
                Card(Rank.TWO, Suit.HEARTS),
                Card(Rank.SEVEN, Suit.CLUBS),
                Card(Rank.JACK, Suit.DIAMONDS),
            ],
            turn_card=Card(Rank.QUEEN, Suit.SPADES),
            river_card=Card(Rank.THREE, Suit.HEARTS),
            preflop_actions=[
                HandAction("hero", PlayerAction.RAISE, 60),
                HandAction("villain", PlayerAction.CALL, 40),
            ],
            flop_actions=[
                HandAction("hero", PlayerAction.BET, 100),
                HandAction("villain", PlayerAction.CALL, 100),
            ],
            pot=320,
            reached_showdown=True,
        )

        d = hand_history_to_dict(hand)
        restored = hand_history_from_dict(d)

        assert restored.hand_id == "67890"
        assert restored.table_name == "Test Table"
        assert restored.timestamp == datetime(2024, 1, 15, 12, 30, 0)
        assert restored.small_blind == 10
        assert restored.big_blind == 20
        assert len(restored.players) == 2
        assert restored.button_seat == 0
        assert len(restored.hero_cards) == 2
        assert len(restored.flop_cards) == 3
        assert restored.turn_card.rank == Rank.QUEEN
        assert restored.river_card.rank == Rank.THREE
        assert len(restored.preflop_actions) == 2
        assert len(restored.flop_actions) == 2
        assert restored.pot == 320
        assert restored.reached_showdown is True


class TestFileIO:
    """Tests for file I/O functions."""

    def test_save_load_game_states(self):
        """GameState list saves and loads from file."""
        states = [
            GameState(
                pot=60,
                hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
                street=Street.PREFLOP,
                frame_number=0,
            ),
            GameState(
                pot=120,
                hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
                community_cards=[
                    Card(Rank.TWO, Suit.HEARTS),
                    Card(Rank.SEVEN, Suit.CLUBS),
                    Card(Rank.JACK, Suit.DIAMONDS),
                ],
                street=Street.FLOP,
                frame_number=100,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_game_states(states, path)
            loaded = load_game_states(path)

            assert len(loaded) == 2
            assert loaded[0].pot == 60
            assert loaded[0].street == Street.PREFLOP
            assert loaded[1].pot == 120
            assert loaded[1].street == Street.FLOP
            assert len(loaded[1].community_cards) == 3
        finally:
            path.unlink()

    def test_save_load_hand_history(self):
        """HandHistory saves and loads from file."""
        hand = HandHistory(
            hand_id="test123",
            players=[
                PlayerInfo(0, "hero", 2000, PlayerPosition.HERO, True),
                PlayerInfo(1, "villain", 1500, PlayerPosition.SEAT_1, False),
            ],
            hero_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)],
            pot=300,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_hand_history(hand, path)
            loaded = load_hand_history(path)

            assert loaded.hand_id == "test123"
            assert len(loaded.players) == 2
            assert len(loaded.hero_cards) == 2
            assert loaded.pot == 300
        finally:
            path.unlink()

    def test_load_game_states_wrong_type(self):
        """Loading wrong file type raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 1, "type": "hand_history", "hand": {}}, f)
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="expected 'game_states'"):
                load_game_states(path)
        finally:
            path.unlink()

    def test_load_hand_history_wrong_type(self):
        """Loading wrong file type raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 1, "type": "game_states", "states": []}, f)
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="expected 'hand_history'"):
                load_hand_history(path)
        finally:
            path.unlink()

    def test_json_is_readable(self):
        """Saved JSON is human-readable (indented)."""
        states = [GameState(pot=60, street=Street.PREFLOP)]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_game_states(states, path)
            content = path.read_text()

            # Check for indentation (pretty printed)
            assert "\n  " in content
        finally:
            path.unlink()
