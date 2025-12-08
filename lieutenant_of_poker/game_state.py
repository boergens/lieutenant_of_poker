"""
Game state aggregator for Governor of Poker.

Combines outputs from all detection modules into a unified GameState object.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Set

import numpy as np

from lieutenant_of_poker.table_regions import (
    TableRegionDetector,
    detect_table_regions,
    NUM_PLAYERS,
    HERO,
)
from lieutenant_of_poker.chip_ocr import extract_pot, extract_player_chips
from lieutenant_of_poker.action_detector import (
    PlayerAction,
    DetectedAction,
)


class Street(Enum):
    """Poker hand streets/stages."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    UNKNOWN = auto()


@dataclass
class PlayerState:
    """State of a single player."""
    position: int  # Seat index 0-4 (HERO = 4)
    name: Optional[str] = None
    chips: Optional[int] = None
    cards: List[str] = field(default_factory=list)
    last_action: Optional[DetectedAction] = None
    is_active: bool = True
    is_dealer: bool = False


@dataclass
class GameState:
    """Complete state of a poker game at a moment in time."""
    # Cards (strings like "Ah", "Kc")
    community_cards: List[str] = field(default_factory=list)
    hero_cards: List[str] = field(default_factory=list)

    # Money
    pot: Optional[int] = None
    hero_chips: Optional[int] = None

    # Players (keyed by seat index 0-4)
    players: Dict[int, PlayerState] = field(default_factory=dict)

    # Game phase
    street: Street = Street.UNKNOWN

    # Metadata
    frame_number: Optional[int] = None
    timestamp_ms: Optional[float] = None
    rejected: bool = False  # True if this state failed validation

    def __str__(self) -> str:
        parts = []

        # Street
        parts.append(f"Street: {self.street.name}")

        # Pot
        if self.pot:
            parts.append(f"Pot: {self.pot:,}")

        # Community cards
        if self.community_cards:
            cards_str = " ".join(str(c) for c in self.community_cards)
            parts.append(f"Board: {cards_str}")

        # Hero cards
        if self.hero_cards:
            cards_str = " ".join(str(c) for c in self.hero_cards)
            parts.append(f"Hero: {cards_str}")

        # Hero chips
        if self.hero_chips:
            parts.append(f"Hero chips: {self.hero_chips:,}")

        return " | ".join(parts)

    @property
    def num_community_cards(self) -> int:
        """Get number of community cards dealt."""
        return len(self.community_cards)

    def determine_street(self) -> Street:
        """Determine the current street based on community cards."""
        n = self.num_community_cards
        if n == 0:
            return Street.PREFLOP
        elif n == 3:
            return Street.FLOP
        elif n == 4:
            return Street.TURN
        elif n == 5:
            return Street.RIVER
        else:
            return Street.UNKNOWN


class GameStateExtractor:
    """Extracts complete game state from video frames."""

    def __init__(self):
        """Initialize the game state extractor."""
        pass

    def extract(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ) -> GameState:
        """
        Extract complete game state from a video frame.

        Args:
            frame: BGR image frame from the game.
            frame_number: Optional frame number for tracking.
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            GameState object with all detected information.
        """
        # Create region detector for this frame
        region_detector = detect_table_regions(frame)

        # Initialize game state
        state = GameState(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )

        # Extract cards directly from frame
        from .card_matcher import match_hero_cards, match_community_cards
        state.community_cards = [c for c in match_community_cards(frame) if c]
        state.hero_cards = [c for c in match_hero_cards(frame) if c]

        state.pot = extract_pot(frame, region_detector)

        # Process all seats by position
        for position in range(NUM_PLAYERS):
            chips = extract_player_chips(frame, region_detector, position)
            if chips is not None:
                state.players[position] = PlayerState(position=position, chips=chips)
        if HERO in state.players:
            state.hero_chips = state.players[HERO].chips

        state.street = state.determine_street()

        return state


def extract_game_state(
    frame: np.ndarray,
    frame_number: Optional[int] = None,
    timestamp_ms: Optional[float] = None,
    active_seats: Optional[Set[int]] = None,
) -> GameState:
    """
    Convenience function to extract game state from a frame.

    Args:
        frame: BGR image frame from the game.
        frame_number: Optional frame number.
        timestamp_ms: Optional timestamp in milliseconds.
        active_seats: Optional set of seat indices to process.

    Returns:
        GameState object with detected information.
    """
    extractor = GameStateExtractor()
    return extractor.extract(frame, frame_number, timestamp_ms, active_seats)
