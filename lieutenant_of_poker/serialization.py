"""
Serialization utilities for GameState and HandHistory.

Enables saving/loading game state data to JSON for testing and debugging.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lieutenant_of_poker.game_state import GameState, PlayerState, Street
from lieutenant_of_poker.hand_history import HandHistory, HandAction, PlayerInfo
from lieutenant_of_poker.card_detector import Card, Rank, Suit
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

def card_to_dict(card: Card) -> Dict[str, str]:
    """Convert a Card to a dictionary."""
    return {
        "rank": card.rank.value,
        "suit": card.suit.value,
    }


def card_from_dict(d: Dict[str, str]) -> Card:
    """Convert a dictionary to a Card."""
    rank = next(r for r in Rank if r.value == d["rank"])
    suit = next(s for s in Suit if s.value == d["suit"])
    return Card(rank=rank, suit=suit)


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
        "cards": [card_to_dict(c) for c in state.cards],
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
        cards=[card_from_dict(c) for c in d.get("cards", [])],
        last_action=detected_action_from_dict(d["last_action"]) if d.get("last_action") else None,
        is_active=d.get("is_active", True),
        is_dealer=d.get("is_dealer", False),
    )


# --- GameState Serialization ---

def game_state_to_dict(state: GameState) -> Dict[str, Any]:
    """Convert a GameState to a dictionary."""
    return {
        "community_cards": [card_to_dict(c) for c in state.community_cards],
        "hero_cards": [card_to_dict(c) for c in state.hero_cards],
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
        community_cards=[card_from_dict(c) for c in d.get("community_cards", [])],
        hero_cards=[card_from_dict(c) for c in d.get("hero_cards", [])],
        pot=d.get("pot"),
        hero_chips=d.get("hero_chips"),
        players=players,
        street=Street[d.get("street", "UNKNOWN")],
        frame_number=d.get("frame_number"),
        timestamp_ms=d.get("timestamp_ms"),
        rejected=d.get("rejected", False),
    )


# --- HandAction Serialization ---

def hand_action_to_dict(action: HandAction) -> Dict[str, Any]:
    """Convert a HandAction to a dictionary."""
    return {
        "player_name": action.player_name,
        "action": action.action.name,
        "amount": action.amount,
    }


def hand_action_from_dict(d: Dict[str, Any]) -> HandAction:
    """Convert a dictionary to a HandAction."""
    return HandAction(
        player_name=d["player_name"],
        action=PlayerAction[d["action"]],
        amount=d.get("amount"),
    )


# --- PlayerInfo Serialization ---

def player_info_to_dict(info: PlayerInfo) -> Dict[str, Any]:
    """Convert a PlayerInfo to a dictionary."""
    return {
        "seat": info.seat,
        "name": info.name,
        "chips": info.chips,
        "position": str(info.position),
        "is_hero": info.is_hero,
    }


def player_info_from_dict(d: Dict[str, Any]) -> PlayerInfo:
    """Convert a dictionary to a PlayerInfo."""
    return PlayerInfo(
        seat=d["seat"],
        name=d["name"],
        chips=d["chips"],
        position=_position_from_str(d["position"]),
        is_hero=d.get("is_hero", False),
    )


# --- HandHistory Serialization ---

def hand_history_to_dict(hand: HandHistory) -> Dict[str, Any]:
    """Convert a HandHistory to a dictionary."""
    return {
        "hand_id": hand.hand_id,
        "table_name": hand.table_name,
        "timestamp": hand.timestamp.isoformat(),
        "small_blind": hand.small_blind,
        "big_blind": hand.big_blind,
        "players": [player_info_to_dict(p) for p in hand.players],
        "button_seat": hand.button_seat,
        "sb_seat": hand.sb_seat,
        "bb_seat": hand.bb_seat,
        "hero_cards": [card_to_dict(c) for c in hand.hero_cards],
        "flop_cards": [card_to_dict(c) for c in hand.flop_cards],
        "turn_card": card_to_dict(hand.turn_card) if hand.turn_card else None,
        "river_card": card_to_dict(hand.river_card) if hand.river_card else None,
        "preflop_actions": [hand_action_to_dict(a) for a in hand.preflop_actions],
        "flop_actions": [hand_action_to_dict(a) for a in hand.flop_actions],
        "turn_actions": [hand_action_to_dict(a) for a in hand.turn_actions],
        "river_actions": [hand_action_to_dict(a) for a in hand.river_actions],
        "pot": hand.pot,
        "hero_went_all_in": hand.hero_went_all_in,
        "hero_folded": hand.hero_folded,
        "reached_showdown": hand.reached_showdown,
        "uncalled_bet": hand.uncalled_bet,
        "uncalled_bet_player": hand.uncalled_bet_player,
    }


def hand_history_from_dict(d: Dict[str, Any]) -> HandHistory:
    """Convert a dictionary to a HandHistory."""
    return HandHistory(
        hand_id=d["hand_id"],
        table_name=d.get("table_name", "Governor of Poker"),
        timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else datetime.now(),
        small_blind=d.get("small_blind", 10),
        big_blind=d.get("big_blind", 20),
        players=[player_info_from_dict(p) for p in d.get("players", [])],
        button_seat=d.get("button_seat", 0),
        sb_seat=d.get("sb_seat", 0),
        bb_seat=d.get("bb_seat", 0),
        hero_cards=[card_from_dict(c) for c in d.get("hero_cards", [])],
        flop_cards=[card_from_dict(c) for c in d.get("flop_cards", [])],
        turn_card=card_from_dict(d["turn_card"]) if d.get("turn_card") else None,
        river_card=card_from_dict(d["river_card"]) if d.get("river_card") else None,
        preflop_actions=[hand_action_from_dict(a) for a in d.get("preflop_actions", [])],
        flop_actions=[hand_action_from_dict(a) for a in d.get("flop_actions", [])],
        turn_actions=[hand_action_from_dict(a) for a in d.get("turn_actions", [])],
        river_actions=[hand_action_from_dict(a) for a in d.get("river_actions", [])],
        pot=d.get("pot", 0),
        hero_went_all_in=d.get("hero_went_all_in", False),
        hero_folded=d.get("hero_folded", False),
        reached_showdown=d.get("reached_showdown", False),
        uncalled_bet=d.get("uncalled_bet", 0),
        uncalled_bet_player=d.get("uncalled_bet_player"),
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


def save_hand_history(hand: HandHistory, path: Union[str, Path]) -> None:
    """
    Save a HandHistory object to a JSON file.

    Args:
        hand: HandHistory object to save.
        path: Output file path.
    """
    data = {
        "version": 1,
        "type": "hand_history",
        "hand": hand_history_to_dict(hand),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_hand_history(path: Union[str, Path]) -> HandHistory:
    """
    Load a HandHistory object from a JSON file.

    Args:
        path: Input file path.

    Returns:
        HandHistory object.
    """
    with open(path) as f:
        data = json.load(f)

    if data.get("type") != "hand_history":
        raise ValueError(f"Invalid file type: expected 'hand_history', got '{data.get('type')}'")

    return hand_history_from_dict(data["hand"])
