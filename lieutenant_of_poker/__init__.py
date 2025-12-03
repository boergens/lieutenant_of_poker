"""
Lieutenant of Poker - Video Analysis for Governor of Poker

A Python library for analyzing screencap videos of Governor of Poker,
extracting game state information through computer vision and OCR.
"""

__version__ = "0.1.0"
__author__ = "Kevin"

# Core modules
from lieutenant_of_poker.frame_extractor import (
    VideoFrameExtractor,
    FrameInfo,
    extract_frame_to_file,
)
from lieutenant_of_poker.table_regions import (
    TableRegionDetector,
    Region,
    PlayerPosition,
    PlayerRegions,
    detect_table_regions,
)
from lieutenant_of_poker.card_detector import (
    Card,
    Suit,
    Rank,
    CardDetector,
    detect_cards_in_region,
)
from lieutenant_of_poker.chip_ocr import (
    extract_pot,
    extract_player_chips,
)
from lieutenant_of_poker.action_detector import (
    PlayerAction,
    DetectedAction,
)
from lieutenant_of_poker.game_state import (
    Street,
    PlayerState,
    GameState,
    GameStateExtractor,
    extract_game_state,
)
from lieutenant_of_poker.hand_history import (
    HandAction,
    PlayerInfo,
    HandHistory,
    HandReconstructor,
)
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.pokerstars_export import export_pokerstars

__all__ = [
    # Frame extraction
    "VideoFrameExtractor",
    "FrameInfo",
    "extract_frame_to_file",
    # Table regions
    "TableRegionDetector",
    "Region",
    "PlayerPosition",
    "PlayerRegions",
    "detect_table_regions",
    # Card detection
    "Card",
    "Suit",
    "Rank",
    "CardDetector",
    "detect_cards_in_region",
    # Chip OCR
    "extract_pot",
    "extract_player_chips",
    # Action types
    "PlayerAction",
    "DetectedAction",
    # Game state
    "Street",
    "PlayerState",
    "GameState",
    "GameStateExtractor",
    "extract_game_state",
    # Hand history
    "HandAction",
    "PlayerInfo",
    "HandHistory",
    "HandReconstructor",
    "export_snowie",
    "export_pokerstars",
]
