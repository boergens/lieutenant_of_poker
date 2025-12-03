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

    def add_folds_for_street(
        actions_list: List[HandAction],
        contributions: Dict[str, int],
        active: List[str],
        acting_order: List[str],
    ) -> List[str]:
        """Add fold actions for players who contributed less than max."""
        if not contributions:
            return active
        max_contrib = max(contributions.values())
        for name in acting_order:
            if name in active and contributions.get(name, 0) < max_contrib:
                actions_list.append(HandAction(name, PlayerAction.FOLD, 0))
                active = [p for p in active if p != name]
        return active

    # Add blind posts as actions at start of preflop
    sb_name = players[sb_idx]
    bb_name = players[bb_idx]
    raw_actions[Street.PREFLOP].insert(0, HandAction(bb_name, PlayerAction.BET, big_blind))
    raw_actions[Street.PREFLOP].insert(0, HandAction(sb_name, PlayerAction.BET, small_blind))

    # Map streets to their action lists and "reached" checks
    street_actions = {
        Street.PREFLOP: hand.preflop_actions,
        Street.FLOP: hand.flop_actions,
        Street.TURN: hand.turn_actions,
        Street.RIVER: hand.river_actions,
    }

    def street_reached(street: Street) -> bool:
        if street == Street.PREFLOP:
            return True
        elif street == Street.FLOP:
            return hand.flop_cards is not None
        elif street == Street.TURN:
            return hand.turn_card is not None
        elif street == Street.RIVER:
            return hand.river_card is not None
        return False

    streets = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]

    for idx, street in enumerate(streets):
        if not street_reached(street) or len(players_active) <= 1:
            break

        actions_list = street_actions[street]
        next_street = streets[idx + 1] if idx + 1 < len(streets) else None

        # First active player from SB position acts first
        for i in range(num_players):
            candidate = players[(sb_idx + i) % num_players]
            if candidate in players_active:
                players_onthespot = rotate_to_first(players_active, candidate)
                break

        contributions: Dict[str, int] = {}
        players_acted: List[str] = []

        for action in raw_actions[street]:
            # Post-flop: add checks for skipped players before a bet
            if street != Street.PREFLOP and action.action == PlayerAction.BET:
                for name in players_onthespot:
                    if name == action.player_name:
                        break
                    if name not in players_acted:
                        actions_list.append(HandAction(name, PlayerAction.CHECK, 0))
                        players_acted.append(name)

            actions_list.append(action)
            players_acted.append(action.player_name)

            if action.action == PlayerAction.FOLD:
                players_active = [p for p in players_active if p != action.player_name]
            elif action.amount:
                contributions[action.player_name] = (
                    contributions.get(action.player_name, 0) + action.amount
                )

        # Handle end-of-street logic
        if next_street and street_reached(next_street):
            # Street completed normally - add checks if no actions recorded
            if street != Street.PREFLOP and not raw_actions[street]:
                for name in players_onthespot:
                    actions_list.append(HandAction(name, PlayerAction.CHECK, 0))
        else:
            # Hand ended on this street - check for folds from unmatched contributions
            players_active = add_folds_for_street(
                actions_list, contributions, players_active, players_onthespot
            )

    # Clean up preflop: remove the synthetic blind actions we added
    hand.preflop_actions = hand.preflop_actions[2:]

    # BB option: if BB limped through (only posted blind, no other action), add check
    bb_acted = any(a.player_name == bb_name for a in hand.preflop_actions)
    if not bb_acted and bb_name in players_active and hand.flop_cards:
        hand.preflop_actions.append(HandAction(bb_name, PlayerAction.CHECK, 0))

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
