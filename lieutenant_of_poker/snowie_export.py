"""
Export game states to PokerSnowie format.

Converts a sequence of GameState transitions into a hand history
that PokerSnowie can analyze.
"""

import io
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TextIO

from .game_state import GameState, Street
from .table_regions import PlayerPosition
from .card_detector import Card


@dataclass
class PlayerAction:
    """A player action inferred from chip changes."""
    player_name: str
    action_type: str  # "call_check", "raise_bet", "folds"
    amount: int
    timestamp_ms: float


def states_to_snowie(
    states: List[GameState],
    hero_name: str = "hero",
    button_pos: int = 0,
) -> str:
    """
    Convert a sequence of GameStates to PokerSnowie format.

    Args:
        states: List of GameState objects from a single hand.
        hero_name: Name to use for the hero player.
        button_pos: Button position (0=SEAT_1, 1=SEAT_2, 2=SEAT_3, 3=SEAT_4, 4=hero).

    Returns:
        String in PokerSnowie/Freezeout format.
    """
    if not states:
        return ""

    # Infer blinds from initial pot (assuming SB:BB = 1:2)
    initial_pot = states[0].pot or 60
    small_blind = initial_pot // 3
    big_blind = small_blind * 2

    output = io.StringIO()
    _write_snowie(output, states, hero_name, small_blind, big_blind, button_pos)
    return output.getvalue()


def _write_snowie(
    f: TextIO,
    states: List[GameState],
    hero_name: str,
    small_blind: int,
    big_blind: int,
    button_pos: int,
) -> None:
    """Write states in PokerSnowie format."""
    initial = states[0]
    final = states[-1]

    # Generate game ID from timestamp
    game_id = str(abs(hash(str(initial.timestamp_ms))) % 100000000)
    timestamp = datetime.now()

    # Canonical seat order (clockwise around table)
    seat_order = [
        PlayerPosition.SEAT_1,
        PlayerPosition.SEAT_2,
        PlayerPosition.SEAT_3,
        PlayerPosition.SEAT_4,
        PlayerPosition.HERO,
    ]

    # Build player info in proper seat order
    players = []  # List of (seat_index, name, chips, position)
    seat_to_name = {}
    for i, pos in enumerate(seat_order):
        if pos in initial.players:
            player = initial.players[pos]
            name = hero_name if pos == PlayerPosition.HERO else pos.name
            chips = player.chips or 2000
            players.append((i, name, chips, pos))
            seat_to_name[pos] = name

    # Header
    f.write("GameStart\n")
    f.write("PokerClient: ExportFormat\n")
    f.write(f"Date: {timestamp.strftime('%d/%m/%Y')}\n")
    f.write("TimeZone: GMT\n")
    f.write(f"Time: {timestamp.strftime('%H:%M:%S')}\n")
    f.write(f"GameId:{game_id}\n")
    f.write("GameType:NoLimit\n")
    f.write("GameCurrency: $\n")
    f.write(f"SmallBlindStake: {small_blind}\n")
    f.write(f"BigBlindStake: {big_blind}\n")
    f.write("AnteStake: 0\n")
    f.write("TableName: Governor of Poker\n")
    f.write(f"Max number of players: {len(players)}\n")
    f.write(f"MyPlayerName: {hero_name}\n")
    f.write(f"DealerPosition: {button_pos}\n")

    # Seats
    for seat_idx, name, chips, _pos in players:
        f.write(f"Seat {seat_idx} {name} {chips}\n")

    # Blinds (SB is button+1, BB is button+2)
    if len(players) >= 2:
        sb_idx = (button_pos + 1) % len(players)
        bb_idx = (button_pos + 2) % len(players)
        f.write(f"SmallBlind: {players[sb_idx][1]} {small_blind}\n")
        f.write(f"BigBlind: {players[bb_idx][1]} {big_blind}\n")

    # Hero's hole cards
    hero_cards = None
    for state in states:
        if state.hero_cards:
            hero_cards = state.hero_cards
            break
    if hero_cards:
        f.write(f"Dealt Cards: {_format_hole_cards(hero_cards)}\n")

    # Track street and write actions
    current_street = Street.PREFLOP
    prev_state = initial

    for state in states[1:]:
        # Check for street change
        new_street = state.street
        if new_street != current_street and new_street != Street.UNKNOWN:
            # Write community cards for new street
            if new_street == Street.FLOP and len(state.community_cards) >= 3:
                cards = state.community_cards[:3]
                f.write(f"FLOP Community Cards:{_format_community_cards(cards)}\n")
            elif new_street == Street.TURN and len(state.community_cards) >= 4:
                cards = state.community_cards[:4]
                f.write(f"TURN Community Cards:{_format_community_cards(cards)}\n")
            elif new_street == Street.RIVER and len(state.community_cards) >= 5:
                cards = state.community_cards[:5]
                f.write(f"RIVER Community Cards:{_format_community_cards(cards)}\n")
            current_street = new_street

        # Detect actions from chip changes
        pot_change = (state.pot or 0) - (prev_state.pot or 0)
        if pot_change > 0:
            # Find who put in chips
            for pos in state.players:
                prev_player = prev_state.players.get(pos)
                curr_player = state.players.get(pos)
                if prev_player and curr_player:
                    prev_chips = prev_player.chips or 0
                    curr_chips = curr_player.chips or 0
                    chip_change = prev_chips - curr_chips
                    if chip_change > 0:
                        player_name = seat_to_name.get(pos, f"Player_{pos.name}")
                        # Infer action type (simplified: all non-zero are call_check)
                        f.write(f"Move: {player_name} call_check {chip_change}\n")

        prev_state = state

    # Winner (hero wins if they have more chips at end than start)
    hero_final_chips = None
    hero_initial_chips = None
    for pos, player in final.players.items():
        if pos == PlayerPosition.HERO:
            hero_final_chips = player.chips
    for pos, player in initial.players.items():
        if pos == PlayerPosition.HERO:
            hero_initial_chips = player.chips

    if hero_final_chips and hero_initial_chips:
        winnings = hero_final_chips - hero_initial_chips
        if winnings > 0:
            f.write(f"Winner: {hero_name} {winnings:.2f}\n")

    f.write("GameEnd\n")
    f.write("\n")


def _format_hole_cards(cards: List[Card]) -> str:
    """Format hole cards: [JdQs]"""
    if not cards:
        return "[]"
    return "[" + "".join(c.short_name for c in cards) + "]"


def _format_community_cards(cards: List[Card]) -> str:
    """Format community cards: [6s Jh 8c]"""
    if not cards:
        return "[]"
    return "[" + " ".join(c.short_name for c in cards) + "]"
