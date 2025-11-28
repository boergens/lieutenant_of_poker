"""
Live state tracker for monitoring game state across frames.

Tracks hand boundaries, detects street changes, and maintains
game state history for rule evaluation.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import numpy as np

from .frame_extractor import FrameInfo
from .game_state import GameState, GameStateExtractor, Street
from .action_detector import PlayerAction, DetectedAction
from .card_detector import Card
from .table_regions import PlayerPosition


class GameEvent(Enum):
    """Events detected during game state tracking."""

    NEW_HAND = auto()  # A new hand has started
    STREET_CHANGE = auto()  # Moved to next street (flop, turn, river)
    ACTION_DETECTED = auto()  # A player action was detected
    HERO_TURN = auto()  # It's now hero's turn to act
    HERO_ACTED = auto()  # Hero has made an action
    HAND_COMPLETE = auto()  # Hand has ended


@dataclass
class TrackedAction:
    """An action tracked during a hand."""

    position: PlayerPosition
    action: PlayerAction
    amount: Optional[int] = None
    street: Street = Street.UNKNOWN
    timestamp_ms: float = 0.0

    def __str__(self) -> str:
        if self.amount:
            return f"{self.position.name}: {self.action.name} {self.amount}"
        return f"{self.position.name}: {self.action.name}"


@dataclass
class TrackedHand:
    """A poker hand being tracked across frames."""

    hand_id: str
    start_time: datetime = field(default_factory=datetime.now)
    states: list[GameState] = field(default_factory=list)
    actions: list[TrackedAction] = field(default_factory=list)
    hero_cards: list[Card] = field(default_factory=list)
    community_cards: list[Card] = field(default_factory=list)
    current_street: Street = Street.PREFLOP
    pot: int = 0
    hero_chips: int = 0
    big_blind: int = 100  # Default, should be detected
    is_complete: bool = False

    @property
    def hero_stack_bb(self) -> float:
        """Get hero stack in big blinds."""
        if self.big_blind <= 0:
            return 0.0
        return self.hero_chips / self.big_blind

    def add_state(self, state: GameState) -> None:
        """Add a state snapshot to the hand."""
        self.states.append(state)

        # Update hand info from state
        if state.hero_cards:
            self.hero_cards = state.hero_cards
        if state.community_cards:
            self.community_cards = state.community_cards
        if state.pot:
            self.pot = state.pot
        if state.hero_chips:
            self.hero_chips = state.hero_chips

        self.current_street = state.street


@dataclass
class StateUpdate:
    """Result of processing a frame."""

    state: GameState
    events: list[GameEvent] = field(default_factory=list)
    new_actions: list[TrackedAction] = field(default_factory=list)

    @property
    def is_new_hand(self) -> bool:
        return GameEvent.NEW_HAND in self.events

    @property
    def is_hero_turn(self) -> bool:
        return GameEvent.HERO_TURN in self.events

    @property
    def street_changed(self) -> bool:
        return GameEvent.STREET_CHANGE in self.events


class LiveStateTracker:
    """
    Tracks game state across frames and detects hand boundaries.

    This class wraps the GameStateExtractor and adds temporal tracking
    to detect when hands start/end, when streets change, and when
    it's the hero's turn to act.
    """

    def __init__(self, extractor: Optional[GameStateExtractor] = None):
        """
        Initialize the state tracker.

        Args:
            extractor: Optional GameStateExtractor to use.
                      Creates one if not provided.
        """
        self.extractor = extractor or GameStateExtractor()
        self.current_hand: Optional[TrackedHand] = None
        self.hand_history: list[TrackedHand] = []
        self._prev_state: Optional[GameState] = None
        self._prev_hero_turn: bool = False

    def process_frame(self, frame: FrameInfo) -> StateUpdate:
        """
        Process a captured frame and track state changes.

        Args:
            frame: The captured frame to process.

        Returns:
            StateUpdate with current state and detected events.
        """
        # Extract game state from frame
        state = self.extractor.extract(
            frame.image,
            frame_number=frame.frame_number,
            timestamp_ms=frame.timestamp_ms,
        )

        events: list[GameEvent] = []
        new_actions: list[TrackedAction] = []

        # Check for new hand
        if self._is_new_hand(self._prev_state, state):
            events.append(GameEvent.NEW_HAND)
            self._start_new_hand()

        # Ensure we have a current hand
        if self.current_hand is None:
            self._start_new_hand()

        # Check for street change
        if self._prev_state and state.street != self._prev_state.street:
            if state.street != Street.UNKNOWN:
                events.append(GameEvent.STREET_CHANGE)

        # Track actions
        new_actions = self._track_actions(self._prev_state, state)
        if new_actions:
            events.append(GameEvent.ACTION_DETECTED)
            self.current_hand.actions.extend(new_actions)

        # Check if it's hero's turn
        is_hero_turn = self._is_hero_turn(state)
        if is_hero_turn and not self._prev_hero_turn:
            events.append(GameEvent.HERO_TURN)

        # Check if hero acted
        hero_action = self._detect_hero_action(new_actions)
        if hero_action:
            events.append(GameEvent.HERO_ACTED)

        # Update current hand
        self.current_hand.add_state(state)

        # Save state for next frame
        self._prev_state = state
        self._prev_hero_turn = is_hero_turn

        return StateUpdate(
            state=state,
            events=events,
            new_actions=new_actions,
        )

    def _is_new_hand(
        self, prev: Optional[GameState], curr: GameState
    ) -> bool:
        """
        Detect if a new hand has started.

        Signals:
        - Community cards went from 5 to 0
        - Community cards went from >0 to 0
        - Hero cards changed (after not being empty)
        - Pot dramatically decreased
        """
        if prev is None:
            return True

        # Community cards reset
        if prev.num_community_cards > 0 and curr.num_community_cards == 0:
            return True

        # Community cards went from 5 (river/showdown) to 0
        if prev.num_community_cards == 5 and curr.num_community_cards == 0:
            return True

        # Hero cards changed (new deal)
        if prev.hero_cards and curr.hero_cards:
            prev_cards = set(str(c) for c in prev.hero_cards)
            curr_cards = set(str(c) for c in curr.hero_cards)
            if prev_cards != curr_cards:
                return True

        # Pot dramatically decreased (new hand pot is small)
        if prev.pot and curr.pot:
            if curr.pot < prev.pot * 0.2:  # Pot dropped by 80%+
                return True

        return False

    def _is_hero_turn(self, state: GameState) -> bool:
        """
        Detect if it's hero's turn to act.

        This is determined by checking if action buttons are visible
        or if hero's position shows as active.
        """
        # Check if hero has a pending action indicator
        hero = state.players.get(PlayerPosition.BOTTOM)
        if hero and hero.last_action:
            # If hero's last action is recent and still shown, it might not be their turn
            pass

        # Heuristic: If we see hero cards and the game is in progress,
        # assume hero is still in the hand and may need to act
        if state.hero_cards and state.street != Street.UNKNOWN:
            # This is a simplified check - real implementation would
            # detect action buttons or turn indicators
            return True

        return False

    def _track_actions(
        self, prev: Optional[GameState], curr: GameState
    ) -> list[TrackedAction]:
        """
        Detect new actions that occurred between frames.
        """
        if prev is None:
            return []

        new_actions = []

        for position in PlayerPosition:
            prev_player = prev.players.get(position)
            curr_player = curr.players.get(position)

            if curr_player and curr_player.last_action:
                # Check if this is a new action
                is_new = True
                if prev_player and prev_player.last_action:
                    if (
                        prev_player.last_action.action
                        == curr_player.last_action.action
                    ):
                        is_new = False

                if is_new:
                    new_actions.append(
                        TrackedAction(
                            position=position,
                            action=curr_player.last_action.action,
                            amount=curr_player.last_action.amount,
                            street=curr.street,
                            timestamp_ms=curr.timestamp_ms or 0.0,
                        )
                    )

        return new_actions

    def _detect_hero_action(
        self, new_actions: list[TrackedAction]
    ) -> Optional[TrackedAction]:
        """Check if any of the new actions are from hero."""
        for action in new_actions:
            if action.position == PlayerPosition.BOTTOM:
                return action
        return None

    def _start_new_hand(self) -> None:
        """Start tracking a new hand."""
        # Archive current hand if exists
        if self.current_hand:
            self.current_hand.is_complete = True
            self.hand_history.append(self.current_hand)

        # Create new hand
        hand_id = str(uuid.uuid4())[:8]
        self.current_hand = TrackedHand(hand_id=hand_id)

    def get_current_hand(self) -> Optional[TrackedHand]:
        """Get the currently tracked hand."""
        return self.current_hand

    def get_hand_count(self) -> int:
        """Get number of hands tracked (including current)."""
        count = len(self.hand_history)
        if self.current_hand:
            count += 1
        return count

    def reset(self) -> None:
        """Reset all tracking state."""
        self.current_hand = None
        self.hand_history.clear()
        self._prev_state = None
        self._prev_hero_turn = False
