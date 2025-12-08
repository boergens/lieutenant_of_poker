"""
Serialization utilities for GameState.

Enables saving/loading game state data to JSON for testing and debugging.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from lieutenant_of_poker.game_state import GameState, PlayerState, Street
from lieutenant_of_poker.action_detector import PlayerAction, DetectedAction
from lieutenant_of_poker.table_regions import HERO


# Mapping from old enum names to new seat indices for backwards compatibility
_OLD_POSITION_NAMES = {
    "SEAT_1": 0,
    "SEAT_2": 1,
    "SEAT_3": 2,
    "SEAT_4": 3,
    "HERO": HERO,
}


def _position_from_str(s: str) -> int:
    """Convert position string to seat index. Handles old enum names and new int strings."""
    if s in _OLD_POSITION_NAMES:
        return _OLD_POSITION_NAMES[s]
    return int(s)


# --- Card Serialization ---
# Cards are now simple strings like "Ah", "Kc" so serialization is trivial


# --- DetectedAction Serialization ---

def detected_action_to_dict(action: DetectedAction) -> Dict[str, Any]:
    """Convert a DetectedAction to a dictionary."""
    return {
        "action": action.action.name,
        "amount": action.amount,
        "confidence": action.confidence,
    }


def detected_action_from_dict(d: Dict[str, Any]) -> DetectedAction:
    """Convert a dictionary to a DetectedAction."""
    return DetectedAction(
        action=PlayerAction[d["action"]],
        amount=d.get("amount"),
        confidence=d.get("confidence", 1.0),
    )


# --- PlayerState Serialization ---

def player_state_to_dict(state: PlayerState) -> Dict[str, Any]:
    """Convert a PlayerState to a dictionary."""
    return {
        "position": str(state.position),
        "name": state.name,
        "chips": state.chips,
        "cards": state.cards,  # Already strings like "Ah"
        "last_action": detected_action_to_dict(state.last_action) if state.last_action else None,
        "is_active": state.is_active,
        "is_dealer": state.is_dealer,
    }


def player_state_from_dict(d: Dict[str, Any]) -> PlayerState:
    """Convert a dictionary to a PlayerState."""
    return PlayerState(
        position=_position_from_str(d["position"]),
        name=d.get("name"),
        chips=d.get("chips"),
        cards=d.get("cards", []),  # Already strings like "Ah"
        last_action=detected_action_from_dict(d["last_action"]) if d.get("last_action") else None,
        is_active=d.get("is_active", True),
        is_dealer=d.get("is_dealer", False),
    )


# --- GameState Serialization ---

def game_state_to_dict(state: GameState) -> Dict[str, Any]:
    """Convert a GameState to a dictionary."""
    return {
        "community_cards": state.community_cards,  # Already strings like "Ah"
        "hero_cards": state.hero_cards,  # Already strings like "Ah"
        "pot": state.pot,
        "hero_chips": state.hero_chips,
        "players": {str(pos): player_state_to_dict(ps) for pos, ps in state.players.items()},
        "street": state.street.name,
        "frame_number": state.frame_number,
        "timestamp_ms": state.timestamp_ms,
        "rejected": state.rejected,
    }


def game_state_from_dict(d: Dict[str, Any]) -> GameState:
    """Convert a dictionary to a GameState."""
    players = {}
    for pos_name, ps_dict in d.get("players", {}).items():
        pos = _position_from_str(pos_name)
        players[pos] = player_state_from_dict(ps_dict)

    return GameState(
        community_cards=d.get("community_cards", []),  # Already strings like "Ah"
        hero_cards=d.get("hero_cards", []),  # Already strings like "Ah"
        pot=d.get("pot"),
        hero_chips=d.get("hero_chips"),
        players=players,
        street=Street[d.get("street", "UNKNOWN")],
        frame_number=d.get("frame_number"),
        timestamp_ms=d.get("timestamp_ms"),
        rejected=d.get("rejected", False),
    )


# --- File I/O ---

def save_game_states(states: List[GameState], path: Union[str, Path]) -> None:
    """
    Save a list of GameState objects to a JSON file.

    Args:
        states: List of GameState objects to save.
        path: Output file path.
    """
    data = {
        "version": 1,
        "type": "game_states",
        "states": [game_state_to_dict(s) for s in states],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_game_states(path: Union[str, Path]) -> List[GameState]:
    """
    Load a list of GameState objects from a JSON file.

    Args:
        path: Input file path.

    Returns:
        List of GameState objects.
    """
    with open(path) as f:
        data = json.load(f)

    if data.get("type") != "game_states":
        raise ValueError(f"Invalid file type: expected 'game_states', got '{data.get('type')}'")

    return [game_state_from_dict(s) for s in data.get("states", [])]
