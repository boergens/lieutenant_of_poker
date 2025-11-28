"""Tests for live state tracker module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lieutenant_of_poker.live_state_tracker import (
    GameEvent,
    LiveStateTracker,
    StateUpdate,
    TrackedAction,
    TrackedHand,
)
from lieutenant_of_poker.frame_extractor import FrameInfo
from lieutenant_of_poker.game_state import GameState, GameStateExtractor, Street
from lieutenant_of_poker.action_detector import PlayerAction, DetectedAction
from lieutenant_of_poker.card_detector import Card, Rank, Suit
from lieutenant_of_poker.table_regions import PlayerPosition


class TestTrackedAction:
    """Tests for TrackedAction dataclass."""

    def test_creation(self):
        """Test creating a tracked action."""
        action = TrackedAction(
            position=PlayerPosition.HERO,
            action=PlayerAction.RAISE,
            amount=100,
            street=Street.PREFLOP,
            timestamp_ms=1000.0,
        )
        assert action.position == PlayerPosition.HERO
        assert action.action == PlayerAction.RAISE
        assert action.amount == 100

    def test_str_with_amount(self):
        """Test string representation with amount."""
        action = TrackedAction(
            position=PlayerPosition.HERO,
            action=PlayerAction.RAISE,
            amount=100,
        )
        result = str(action)
        assert "HERO" in result
        assert "RAISE" in result
        assert "100" in result

    def test_str_without_amount(self):
        """Test string representation without amount."""
        action = TrackedAction(
            position=PlayerPosition.SEAT_2,
            action=PlayerAction.FOLD,
        )
        result = str(action)
        assert "SEAT_2" in result
        assert "FOLD" in result


class TestTrackedHand:
    """Tests for TrackedHand dataclass."""

    def test_hero_stack_bb(self):
        """Test stack in big blinds calculation."""
        hand = TrackedHand(
            hand_id="test",
            hero_chips=1500,
            big_blind=100,
        )
        assert hand.hero_stack_bb == 15.0

    def test_hero_stack_bb_zero_blind(self):
        """Test stack with zero big blind."""
        hand = TrackedHand(
            hand_id="test",
            hero_chips=1000,
            big_blind=0,
        )
        assert hand.hero_stack_bb == 0.0

    def test_add_state(self):
        """Test adding state to hand."""
        hand = TrackedHand(hand_id="test")

        state = GameState(
            hero_cards=[
                Card(Rank.ACE, Suit.SPADES),
                Card(Rank.KING, Suit.SPADES),
            ],
            community_cards=[
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.JACK, Suit.HEARTS),
                Card(Rank.TEN, Suit.HEARTS),
            ],
            pot=500,
            hero_chips=1000,
            street=Street.FLOP,
        )

        hand.add_state(state)

        assert len(hand.states) == 1
        assert len(hand.hero_cards) == 2
        assert len(hand.community_cards) == 3
        assert hand.pot == 500
        assert hand.hero_chips == 1000
        assert hand.current_street == Street.FLOP


class TestStateUpdate:
    """Tests for StateUpdate dataclass."""

    def test_is_new_hand(self):
        """Test is_new_hand property."""
        state = GameState()
        update = StateUpdate(state=state, events=[GameEvent.NEW_HAND])
        assert update.is_new_hand is True

        update = StateUpdate(state=state, events=[])
        assert update.is_new_hand is False

    def test_is_hero_turn(self):
        """Test is_hero_turn property."""
        state = GameState()
        update = StateUpdate(state=state, events=[GameEvent.HERO_TURN])
        assert update.is_hero_turn is True

        update = StateUpdate(state=state, events=[])
        assert update.is_hero_turn is False

    def test_street_changed(self):
        """Test street_changed property."""
        state = GameState()
        update = StateUpdate(state=state, events=[GameEvent.STREET_CHANGE])
        assert update.street_changed is True

        update = StateUpdate(state=state, events=[])
        assert update.street_changed is False


class TestLiveStateTracker:
    """Tests for LiveStateTracker class."""

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock game state extractor."""
        extractor = MagicMock(spec=GameStateExtractor)
        return extractor

    @pytest.fixture
    def tracker(self, mock_extractor):
        """Create a tracker with mock extractor."""
        return LiveStateTracker(extractor=mock_extractor)

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame."""
        return FrameInfo(
            frame_number=0,
            timestamp_ms=0.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

    def test_init_default_extractor(self):
        """Test initialization creates default extractor."""
        tracker = LiveStateTracker()
        assert tracker.extractor is not None
        assert tracker.current_hand is None

    def test_init_custom_extractor(self, mock_extractor):
        """Test initialization with custom extractor."""
        tracker = LiveStateTracker(extractor=mock_extractor)
        assert tracker.extractor == mock_extractor

    def test_process_first_frame_creates_hand(self, tracker, mock_extractor, sample_frame):
        """Test that first frame creates a new hand."""
        mock_extractor.extract.return_value = GameState(street=Street.PREFLOP)

        update = tracker.process_frame(sample_frame)

        assert tracker.current_hand is not None
        assert GameEvent.NEW_HAND in update.events

    def test_process_frame_extracts_state(self, tracker, mock_extractor, sample_frame):
        """Test that frame processing extracts state."""
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            hero_chips=1000,
        )
        mock_extractor.extract.return_value = state

        update = tracker.process_frame(sample_frame)

        assert update.state == state
        mock_extractor.extract.assert_called_once()

    def test_detect_new_hand_community_cards_reset(self, tracker, mock_extractor, sample_frame):
        """Test new hand detection when community cards reset."""
        # First state: river with 5 cards
        state1 = GameState(
            street=Street.RIVER,
            community_cards=[
                Card(Rank.ACE, Suit.SPADES),
                Card(Rank.KING, Suit.SPADES),
                Card(Rank.QUEEN, Suit.SPADES),
                Card(Rank.JACK, Suit.SPADES),
                Card(Rank.TEN, Suit.SPADES),
            ],
        )

        # Second state: preflop with no community cards
        state2 = GameState(
            street=Street.PREFLOP,
            community_cards=[],
        )

        mock_extractor.extract.side_effect = [state1, state2]

        # Process first frame
        tracker.process_frame(sample_frame)
        first_hand = tracker.current_hand

        # Process second frame
        update = tracker.process_frame(sample_frame)

        assert GameEvent.NEW_HAND in update.events
        assert tracker.current_hand != first_hand

    def test_detect_new_hand_hero_cards_change(self, tracker, mock_extractor, sample_frame):
        """Test new hand detection when hero cards change."""
        state1 = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.ACE, Suit.SPADES),
                Card(Rank.KING, Suit.SPADES),
            ],
        )

        state2 = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.SEVEN, Suit.HEARTS),
                Card(Rank.TWO, Suit.CLUBS),
            ],
        )

        mock_extractor.extract.side_effect = [state1, state2]

        tracker.process_frame(sample_frame)
        update = tracker.process_frame(sample_frame)

        assert GameEvent.NEW_HAND in update.events

    def test_detect_new_hand_pot_decrease(self, tracker, mock_extractor, sample_frame):
        """Test new hand detection when pot dramatically decreases."""
        state1 = GameState(street=Street.RIVER, pot=1000)
        state2 = GameState(street=Street.PREFLOP, pot=15)  # 98.5% decrease

        mock_extractor.extract.side_effect = [state1, state2]

        tracker.process_frame(sample_frame)
        update = tracker.process_frame(sample_frame)

        assert GameEvent.NEW_HAND in update.events

    def test_detect_street_change(self, tracker, mock_extractor, sample_frame):
        """Test street change detection."""
        state1 = GameState(street=Street.PREFLOP)
        state2 = GameState(street=Street.FLOP)

        mock_extractor.extract.side_effect = [state1, state2]

        tracker.process_frame(sample_frame)
        update = tracker.process_frame(sample_frame)

        assert GameEvent.STREET_CHANGE in update.events

    def test_track_new_action(self, tracker, mock_extractor, sample_frame):
        """Test action tracking."""
        from lieutenant_of_poker.game_state import PlayerState

        state1 = GameState(
            street=Street.PREFLOP,
            players={
                PlayerPosition.SEAT_2: PlayerState(
                    position=PlayerPosition.SEAT_2,
                    last_action=None,
                )
            },
        )

        state2 = GameState(
            street=Street.PREFLOP,
            players={
                PlayerPosition.SEAT_2: PlayerState(
                    position=PlayerPosition.SEAT_2,
                    last_action=DetectedAction(
                        action=PlayerAction.RAISE,
                        amount=100,
                    ),
                )
            },
        )

        mock_extractor.extract.side_effect = [state1, state2]

        tracker.process_frame(sample_frame)
        update = tracker.process_frame(sample_frame)

        assert GameEvent.ACTION_DETECTED in update.events
        assert len(update.new_actions) == 1
        assert update.new_actions[0].action == PlayerAction.RAISE

    def test_get_current_hand(self, tracker, mock_extractor, sample_frame):
        """Test getting current hand."""
        mock_extractor.extract.return_value = GameState(street=Street.PREFLOP)

        tracker.process_frame(sample_frame)

        hand = tracker.get_current_hand()
        assert hand is not None
        assert hand.hand_id is not None

    def test_get_hand_count(self, tracker, mock_extractor, sample_frame):
        """Test hand counting."""
        # Create states that trigger new hands
        state1 = GameState(street=Street.RIVER, pot=1000)
        state2 = GameState(street=Street.PREFLOP, pot=10)
        state3 = GameState(street=Street.RIVER, pot=2000)
        state4 = GameState(street=Street.PREFLOP, pot=10)

        mock_extractor.extract.side_effect = [state1, state2, state3, state4]

        for _ in range(4):
            tracker.process_frame(sample_frame)

        # Should have 3 hands: 2 completed + 1 current
        assert tracker.get_hand_count() == 3

    def test_reset(self, tracker, mock_extractor, sample_frame):
        """Test resetting the tracker."""
        mock_extractor.extract.return_value = GameState(street=Street.PREFLOP)

        tracker.process_frame(sample_frame)
        assert tracker.current_hand is not None

        tracker.reset()

        assert tracker.current_hand is None
        assert len(tracker.hand_history) == 0

    def test_hand_history_preserved(self, tracker, mock_extractor, sample_frame):
        """Test that hand history is preserved across new hands."""
        state1 = GameState(street=Street.RIVER, pot=1000)
        state2 = GameState(street=Street.PREFLOP, pot=10)

        mock_extractor.extract.side_effect = [state1, state2]

        tracker.process_frame(sample_frame)
        first_hand_id = tracker.current_hand.hand_id

        tracker.process_frame(sample_frame)

        assert len(tracker.hand_history) == 1
        assert tracker.hand_history[0].hand_id == first_hand_id
        assert tracker.hand_history[0].is_complete is True
