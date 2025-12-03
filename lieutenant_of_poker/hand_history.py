"""
Hand history reconstruction from GameState observations.

Shared data structures and reconstruction logic used by all exporters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.card_detector import Card
from lieutenant_of_poker.action_detector import PlayerAction


@dataclass
class HandAction:
    """A single action in a hand."""
    player_name: str
    action: PlayerAction
    amount: Optional[int] = None


@dataclass
class PlayerInfo:
    """Information about a player at the start of the hand."""
    name: str
    chips: int
    position: int  # Original seat index 0-4


@dataclass
class HandHistory:
    """Complete history of a single poker hand."""
    table_name: str = "Governor of Poker"
    timestamp: datetime = field(default_factory=datetime.now)

    small_blind: int = 10
    big_blind: int = 20

    players: List[PlayerInfo] = field(default_factory=list)
    button_seat: int = 0
    sb_seat: int = 0
    bb_seat: int = 0

    hero_cards: List[Card] = field(default_factory=list)
    flop_cards: Optional[List[Card]] = None
    turn_card: Optional[Card] = None
    river_card: Optional[Card] = None

    preflop_actions: List[HandAction] = field(default_factory=list)
    flop_actions: List[HandAction] = field(default_factory=list)
    turn_actions: List[HandAction] = field(default_factory=list)
    river_actions: List[HandAction] = field(default_factory=list)

    pot: int = 0

    # Outcome flags
    hero_went_all_in: bool = False
    hero_folded: bool = False
    opponents_folded: bool = False
    reached_showdown: bool = False


def reconstruct_hand(
    states: List[GameState],
    hero_name: str,
    player_names: Dict[int, str],
    button_pos: int,
) -> Optional[HandHistory]:
    """Reconstruct a HandHistory from a sequence of GameState observations.

    Args:
        states: List of GameState objects representing the hand progression
        hero_name: Name of the hero player (must match a value in player_names)
        player_names: Mapping of seat positions to player names (all active players)
        button_pos: Button position (index into player list)

    Returns:
        HandHistory if reconstruction succeeds, None otherwise
    """
    if not states:
        return None

    initial, final = states[0], states[-1]

    # Derive blinds from initial pot (assumes SB + BB + antes = pot)
    initial_pot = initial.pot or 60
    small_blind = initial_pot // 3
    big_blind = small_blind * 2

    # players: ordered list of names by seat position
    players = [player_names[pos] for pos in sorted(player_names.keys())]
    num_players = len(players)
    if num_players == 0:
        return None

    # Calculate blind positions first (heads-up: button = SB)
    if num_players == 2:
        sb_idx, bb_idx = button_pos, (button_pos + 1) % 2
    else:
        sb_idx = (button_pos + 1) % num_players
        bb_idx = (button_pos + 2) % num_players

    # Build PlayerInfo list and position mapping
    # Add blinds back since initial state shows chips AFTER posting
    player_infos: List[PlayerInfo] = []
    pos_to_name: Dict[int, str] = {}
    for i, pos in enumerate(sorted(player_names.keys())):
        name = player_names[pos]
        chips = initial.players[pos].chips if pos in initial.players else 0
        if i == sb_idx:
            chips += small_blind
        elif i == bb_idx:
            chips += big_blind
        player_infos.append(PlayerInfo(name, chips, pos))
        pos_to_name[pos] = name

    hand = HandHistory(
        small_blind=small_blind,
        big_blind=big_blind,
        players=player_infos,
        button_seat=button_pos,
        sb_seat=sb_idx,
        bb_seat=bb_idx,
        pot=final.pot or 0,
    )

    # Find hero cards from any state that has them
    for state in states:
        if state.hero_cards:
            hand.hero_cards = state.hero_cards
            break

    # Detect actions from chip changes, organized by street
    raw_actions: Dict[Street, List[HandAction]] = {
        Street.PREFLOP: [],
        Street.FLOP: [],
        Street.TURN: [],
        Street.RIVER: [],
    }

    current_street = Street.PREFLOP
    prev_state = states[0]
    current_bet = big_blind

    for state in states[1:]:
        new_street = state.street
        if new_street != current_street and new_street != Street.UNKNOWN:
            if new_street == Street.FLOP and len(state.community_cards) >= 3:
                hand.flop_cards = state.community_cards[:3]
            elif new_street == Street.TURN and len(state.community_cards) >= 4:
                hand.turn_card = state.community_cards[3]
            elif new_street == Street.RIVER and len(state.community_cards) >= 5:
                hand.river_card = state.community_cards[4]
            current_street = new_street
            current_bet = 0

        pot_change = (state.pot or 0) - (prev_state.pot or 0)
        if pot_change > 0:
            for pos in state.players:
                prev_p, curr_p = prev_state.players.get(pos), state.players.get(pos)
                if prev_p and curr_p and pos in pos_to_name:
                    chip_change = (prev_p.chips or 0) - (curr_p.chips or 0)
                    if chip_change > 0:
                        name = pos_to_name[pos]
                        if chip_change > current_bet:
                            action_type = PlayerAction.BET if current_bet == 0 else PlayerAction.RAISE
                        else:
                            action_type = PlayerAction.CALL
                        if chip_change > current_bet:
                            current_bet = chip_change

                        action = HandAction(name, action_type, chip_change)
                        if current_street in raw_actions:
                            raw_actions[current_street].append(action)

        prev_state = state

    # players_active: who's still in the hand (shrinks as players fold)
    players_active = list(players)

    def rotate_to_first(lst: List[str], first: str) -> List[str]:
        """Rotate list so first appears at index 0."""
        if first not in lst:
            return lst
        idx = lst.index(first)
        return lst[idx:] + lst[:idx]

    # ============================================================
    # PREFLOP
    # ============================================================
    # UTG acts first (player after BB), in heads-up SB acts first
    utg_idx = (bb_idx + 1) % num_players
    players_onthespot = rotate_to_first(players_active, players[utg_idx])

    players_acted: List[str] = []
    for action in raw_actions[Street.PREFLOP]:
        hand.preflop_actions.append(action)
        players_acted.append(action.player_name)
        if action.action == PlayerAction.FOLD:
            players_active = [p for p in players_active if p != action.player_name]

    # BB option: if BB hasn't acted and we're going to flop, BB checked
    if hand.flop_cards:
        bb_name = players[bb_idx]
        if bb_name not in players_acted and bb_name in players_active:
            hand.preflop_actions.append(HandAction(bb_name, PlayerAction.CHECK, 0))

    # ============================================================
    # FLOP
    # ============================================================
    if hand.flop_cards and len(players_active) > 1:
        # SB (or first remaining player) acts first post-flop
        players_onthespot = rotate_to_first(players_active, players_active[0])
        for i in range(num_players):
            candidate = players[(sb_idx + i) % num_players]
            if candidate in players_active:
                players_onthespot = rotate_to_first(players_active, candidate)
                break

        players_acted = []
        for action in raw_actions[Street.FLOP]:
            # Add checks for skipped players
            if action.action == PlayerAction.BET:
                for name in players_onthespot:
                    if name == action.player_name:
                        break
                    if name not in players_acted:
                        hand.flop_actions.append(HandAction(name, PlayerAction.CHECK, 0))
                        players_acted.append(name)

            hand.flop_actions.append(action)
            players_acted.append(action.player_name)
            if action.action == PlayerAction.FOLD:
                players_active = [p for p in players_active if p != action.player_name]

        if not raw_actions[Street.FLOP] and hand.turn_card:
            # No actions but we got to turn = everyone checked
            for name in players_onthespot:
                hand.flop_actions.append(HandAction(name, PlayerAction.CHECK, 0))

    # ============================================================
    # TURN
    # ============================================================
    if hand.turn_card and len(players_active) > 1:
        # First active player from SB position acts first
        players_onthespot = list(players_active)
        for i in range(num_players):
            candidate = players[(sb_idx + i) % num_players]
            if candidate in players_active:
                players_onthespot = rotate_to_first(players_active, candidate)
                break

        players_acted = []
        for action in raw_actions[Street.TURN]:
            if action.action == PlayerAction.BET:
                for name in players_onthespot:
                    if name == action.player_name:
                        break
                    if name not in players_acted:
                        hand.turn_actions.append(HandAction(name, PlayerAction.CHECK, 0))
                        players_acted.append(name)

            hand.turn_actions.append(action)
            players_acted.append(action.player_name)
            if action.action == PlayerAction.FOLD:
                players_active = [p for p in players_active if p != action.player_name]

        if not raw_actions[Street.TURN] and hand.river_card:
            for name in players_onthespot:
                hand.turn_actions.append(HandAction(name, PlayerAction.CHECK, 0))

    # ============================================================
    # RIVER
    # ============================================================
    if hand.river_card and len(players_active) > 1:
        # First active player from SB position acts first
        players_onthespot = list(players_active)
        for i in range(num_players):
            candidate = players[(sb_idx + i) % num_players]
            if candidate in players_active:
                players_onthespot = rotate_to_first(players_active, candidate)
                break

        players_acted = []
        for action in raw_actions[Street.RIVER]:
            if action.action == PlayerAction.BET:
                for name in players_onthespot:
                    if name == action.player_name:
                        break
                    if name not in players_acted:
                        hand.river_actions.append(HandAction(name, PlayerAction.CHECK, 0))
                        players_acted.append(name)

            hand.river_actions.append(action)
            players_acted.append(action.player_name)
            if action.action == PlayerAction.FOLD:
                players_active = [p for p in players_active if p != action.player_name]

    # Showdown if multiple players remain after all action
    hand.reached_showdown = len(players_active) > 1

    return hand


# Backwards compatibility alias
class HandReconstructor:
    """Deprecated: Use reconstruct_hand() function instead."""

    def __init__(
        self,
        hero_name: str = "hero",
        player_names: Optional[Dict[int, str]] = None,
    ):
        self.hero_name = hero_name
        self.player_names = player_names or {}

    def reconstruct(
        self,
        states: List[GameState],
        button_pos: Optional[int] = None,
    ) -> Optional[HandHistory]:
        return reconstruct_hand(states, self.hero_name, self.player_names, button_pos)
