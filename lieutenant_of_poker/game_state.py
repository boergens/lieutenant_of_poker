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
    BASE_WIDTH,
    BASE_HEIGHT,
)
from lieutenant_of_poker.card_detector import Card, CardDetector
from lieutenant_of_poker.chip_ocr import ChipOCR, OCRCache
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
    # Uses BASE from table_regions to ensure coordinates match
    TARGET_WIDTH = BASE_WIDTH
    TARGET_HEIGHT = BASE_HEIGHT

    def __init__(self, scale_frames: bool = True):
        """
        Initialize the game state extractor.

        Args:
            scale_frames: If True, scale down frames larger than target resolution.
        """
        self.card_detector = CardDetector()
        self.chip_ocr = ChipOCR()
        self.scale_frames = scale_frames

        # Per-slot OCR caches (pot + each player position)
        self._pot_cache = OCRCache(max_size=3)
        self._chip_caches: Dict[PlayerPosition, OCRCache] = {
            pos: OCRCache(max_size=3) for pos in PlayerPosition
        }

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
            h, w = frame.shape[:2]
            if w > self.TARGET_WIDTH or h > self.TARGET_HEIGHT:
                scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
            for i, slot_img in enumerate(card_slots):
                card = self.card_detector.detect_card(slot_img, slot_index=i)
                if card:
                    cards.append(card)
            state.community_cards = cards
        except Exception:
            pass

        # Extract hero cards from full hero region using calibrated subregions
        try:
            from .card_matcher import match_hero_cards
            hero_region = region_detector.extract_hero_cards(frame)
            hero_cards = match_hero_cards(hero_region)
            # Filter out None results
            state.hero_cards = [c for c in hero_cards if c is not None]
        except Exception:
            pass

        # Extract pot (with cache)
        try:
            pot_region = region_detector.extract_pot(frame)
            found, cached_pot = self._pot_cache.get(pot_region)
            if found:
                state.pot = cached_pot
            else:
                state.pot = self.chip_ocr.extract_pot(pot_region)
                self._pot_cache.put(pot_region, state.pot)
        except Exception:
            pass

        # Extract player states
        for position in PlayerPosition:
            player_state = self._extract_player_state(
                frame, region_detector, position
            )
            state.players[position] = player_state

        # Get hero chips from hero player state
        if PlayerPosition.HERO in state.players:
            state.hero_chips = state.players[PlayerPosition.HERO].chips

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

        # Extract chips (with per-player cache)
        try:
            chip_region = player_regions.name_chip_box.extract(frame)
            cache = self._chip_caches[position]
            found, cached_chips = cache.get(chip_region)
            if found:
                player_state.chips = cached_chips
            else:
                player_state.chips = self.chip_ocr.extract_player_chips(chip_region)
                cache.put(chip_region, player_state.chips)
        except Exception:
            pass

        # Note: Actions are now deduced from chip changes in video_analyzer.py
        # Note: Hero cards are extracted separately via match_hero_cards()

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
