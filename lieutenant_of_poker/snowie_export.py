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
    bb_name = None
    if len(players) >= 2:
        sb_idx = (button_pos + 1) % len(players)
        bb_idx = (button_pos + 2) % len(players)
        bb_name = players[bb_idx][1]
        f.write(f"SmallBlind: {players[sb_idx][1]} {small_blind}\n")
        f.write(f"BigBlind: {bb_name} {big_blind}\n")

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
    street_had_action = False  # Track if any betting happened on current street
    last_aggressor_seat = None  # Track seat index of last player who bet/raised
    current_bet = big_blind  # Current bet to call (starts at BB for preflop)

    for state in states[1:]:
        # Check for street change
        new_street = state.street
        if new_street != current_street and new_street != Street.UNKNOWN:
            # BB check when preflop ends without a raise above BB
            if current_street == Street.PREFLOP and current_bet == big_blind and bb_name:
                f.write(f"Move: {bb_name} call_check 0\n")
            # If no betting action on previous street (post-flop), emit checks for all players
            # Post-flop order starts from SB (button+1)
            elif current_street != Street.PREFLOP and not street_had_action:
                for i in range(len(players)):
                    idx = (sb_idx + i) % len(players)
                    f.write(f"Move: {players[idx][1]} call_check 0\n")

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
            street_had_action = False  # Reset for new street
            last_aggressor_seat = None
            current_bet = 0  # Reset bet for new street

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
                        # Find this player's seat index
                        actor_seat = None
                        for idx, (seat_idx, name, _, p) in enumerate(players):
                            if p == pos:
                                actor_seat = seat_idx
                                break

                        # Post-flop: emit checks for players before this actor (if first bet on street)
                        if current_street != Street.PREFLOP and not street_had_action:
                            for i in range(len(players)):
                                idx = (sb_idx + i) % len(players)
                                if idx == actor_seat:
                                    break
                                f.write(f"Move: {players[idx][1]} call_check 0\n")

                        street_had_action = True
                        last_aggressor_seat = actor_seat

                        # Determine if call or raise
                        if chip_change > current_bet:
                            f.write(f"Move: {player_name} raise_bet {chip_change}\n")
                            current_bet = chip_change
                        else:
                            f.write(f"Move: {player_name} call_check {chip_change}\n")

        prev_state = state

    # Video stops on hero action - emit folds for players after last aggressor, then hero
    last_aggressor_name = None
    last_bet_amount = 0
    if last_aggressor_seat is not None:
        # Find last aggressor's name and bet amount
        for seat_idx, name, _, pos in players:
            if seat_idx == last_aggressor_seat:
                last_aggressor_name = name
        # Fold players between last aggressor and hero (in seat order)
        for seat_idx, name, _, pos in players:
            if seat_idx > last_aggressor_seat and pos != PlayerPosition.HERO:
                f.write(f"Move: {name} folds 0\n")
    f.write(f"Move: {hero_name} folds 0\n")

    # Uncalled bet returns to last aggressor, they win the contested pot
    final_pot = final.pot or 0
    # Find the last bet amount from chip changes
    if len(states) >= 2:
        for pos in states[-1].players:
            prev_player = states[-2].players.get(pos)
            curr_player = states[-1].players.get(pos)
            if prev_player and curr_player:
                chip_diff = (prev_player.chips or 0) - (curr_player.chips or 0)
                if chip_diff > 0:
                    last_bet_amount = chip_diff

    if last_aggressor_name and last_bet_amount > 0:
        f.write(f"Move: {last_aggressor_name} uncalled_bet {last_bet_amount}\n")
        winnings = final_pot - last_bet_amount
        f.write(f"Winner: {last_aggressor_name} {winnings:.2f}\n")

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
