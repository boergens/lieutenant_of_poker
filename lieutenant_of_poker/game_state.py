"""
Game state aggregator for Governor of Poker.

Combines outputs from all detection modules into a unified GameState object.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict

import cv2
import numpy as np

from lieutenant_of_poker.table_regions import (
    TableRegionDetector,
    PlayerPosition,
    detect_table_regions,
)
from lieutenant_of_poker.card_detector import Card, CardDetector
from lieutenant_of_poker.chip_ocr import ChipOCR
from lieutenant_of_poker.action_detector import (
    PlayerAction,
    DetectedAction,
    ActionDetector,
)


class Street(Enum):
    """Poker hand streets/stages."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    UNKNOWN = auto()


@dataclass
class PlayerState:
    """State of a single player."""
    position: PlayerPosition
    name: Optional[str] = None
    chips: Optional[int] = None
    cards: List[Card] = field(default_factory=list)
    last_action: Optional[DetectedAction] = None
    is_active: bool = True
    is_dealer: bool = False


@dataclass
class GameState:
    """Complete state of a poker game at a moment in time."""
    # Cards
    community_cards: List[Card] = field(default_factory=list)
    hero_cards: List[Card] = field(default_factory=list)

    # Money
    pot: Optional[int] = None
    hero_chips: Optional[int] = None

    # Players
    players: Dict[PlayerPosition, PlayerState] = field(default_factory=dict)

    # Game phase
    street: Street = Street.UNKNOWN

    # Metadata
    frame_number: Optional[int] = None
    timestamp_ms: Optional[float] = None

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

    # Target resolution for processing (scale down larger frames)
    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080

    def __init__(self, scale_frames: bool = True):
        """
        Initialize the game state extractor.

        Args:
            scale_frames: If True, scale down frames larger than target resolution.
        """
        self.card_detector = CardDetector()
        self.chip_ocr = ChipOCR()
        self.action_detector = ActionDetector()
        self.scale_frames = scale_frames

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Scale frame down if larger than target resolution."""
        h, w = frame.shape[:2]

        if w <= self.TARGET_WIDTH and h <= self.TARGET_HEIGHT:
            return frame

        # Calculate scale to fit within target
        scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def extract(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp_ms: Optional[float] = None
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
        # Scale down large frames for faster processing
        if self.scale_frames:
            frame = self._scale_frame(frame)

        # Create region detector for this frame
        region_detector = detect_table_regions(frame)

        # Initialize game state
        state = GameState(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )

        # Extract community cards using fixed slots
        try:
            card_slots = region_detector.extract_community_card_slots(frame)
            cards = []
            for slot_img in card_slots:
                card = self.card_detector.detect_card(slot_img)
                if card:
                    cards.append(card)
            state.community_cards = cards
        except Exception:
            pass

        # Extract hero cards using fixed slots
        try:
            hero_slots = region_detector.extract_hero_card_slots(frame)
            cards = []
            for slot_img in hero_slots:
                card = self.card_detector.detect_card(slot_img)
                if card:
                    cards.append(card)
            state.hero_cards = cards
        except Exception:
            pass

        # Extract pot
        try:
            pot_region = region_detector.extract_pot(frame)
            state.pot = self.chip_ocr.extract_pot(pot_region)
        except Exception:
            pass

        # Extract player states
        for position in PlayerPosition:
            player_state = self._extract_player_state(
                frame, region_detector, position
            )
            state.players[position] = player_state

        # Get hero chips from hero player state
        if PlayerPosition.BOTTOM in state.players:
            state.hero_chips = state.players[PlayerPosition.BOTTOM].chips

        # Determine street
        state.street = state.determine_street()

        return state

    def _extract_player_state(
        self,
        frame: np.ndarray,
        region_detector: TableRegionDetector,
        position: PlayerPosition
    ) -> PlayerState:
        """Extract state for a single player."""
        player_regions = region_detector.get_player_region(position)
        player_state = PlayerState(position=position)

        # Extract chips
        try:
            chip_region = player_regions.name_chip_box.extract(frame)
            player_state.chips = self.chip_ocr.extract_player_chips(chip_region)
        except Exception:
            pass

        # Extract last action
        try:
            action_region = player_regions.action_label.extract(frame)
            player_state.last_action = self.action_detector.detect_action_label(
                action_region
            )
        except Exception:
            pass

        # Extract cards (only for hero or during showdown)
        if position == PlayerPosition.BOTTOM and player_regions.cards:
            try:
                card_region = player_regions.cards.extract(frame)
                player_state.cards = self.card_detector.detect_cards(card_region)
            except Exception:
                pass

        return player_state


def extract_game_state(
    frame: np.ndarray,
    frame_number: Optional[int] = None,
    timestamp_ms: Optional[float] = None
) -> GameState:
    """
    Convenience function to extract game state from a frame.

    Args:
        frame: BGR image frame from the game.
        frame_number: Optional frame number.
        timestamp_ms: Optional timestamp in milliseconds.

    Returns:
        GameState object with detected information.
    """
    extractor = GameStateExtractor()
    return extractor.extract(frame, frame_number, timestamp_ms)
