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

    actions: Dict[Street, List[HandAction]] = field(default_factory=lambda: {
        Street.PREFLOP: [],
        Street.FLOP: [],
        Street.TURN: [],
        Street.RIVER: [],
    })

    pot: int = 0

    # Outcome flags
    hero_went_all_in: bool = False
    hero_folded: bool = False
    opponents_folded: bool = False
    reached_showdown: bool = False

    # Winner info
    winner: Optional[str] = None
    payout: int = 0


def reconstruct_hand(
    states: List[GameState],
    players: List[str],
    button_pos: int,
    hero_cards: List[Card],
) -> Optional[HandHistory]:
    """Reconstruct a HandHistory from a sequence of GameState observations.

    Args:
        states: List of GameState objects representing the hand progression
        players: Ordered list of player names by seat position
        button_pos: Button position (index into player list)
        hero_cards: Hero's hole cards

    Returns:
        HandHistory if reconstruction succeeds, None otherwise

    Note:
        The hero is always the last player in the list.
    """
    if not states:
        return None

    initial, final = states[0], states[-1]

    # Derive blinds from initial pot (assumes SB + BB + antes = pot)
    small_blind = (initial.pot or 60) // 3
    big_blind = small_blind * 2

    if not players:
        return None

    # Calculate blind positions first (heads-up: button = SB)
    if len(players) == 2:
        sb_idx, bb_idx = button_pos, (button_pos + 1) % 2
    else:
        sb_idx = (button_pos + 1) % len(players)
        bb_idx = (button_pos + 2) % len(players)

    # Build PlayerInfo list
    # Add blinds back since initial state shows chips AFTER posting
    player_infos: List[PlayerInfo] = []
    for i, name in enumerate(players):
        chips = initial.players[i].chips if i in initial.players else 0
        if i == sb_idx:
            chips += small_blind
        elif i == bb_idx:
            chips += big_blind
        player_infos.append(PlayerInfo(name, chips, i))

    hand = HandHistory(
        small_blind=small_blind,
        big_blind=big_blind,
        players=player_infos,
        button_seat=button_pos,
        sb_seat=sb_idx,
        bb_seat=bb_idx,
        pot=final.pot or 0,
    )

    hand.hero_cards = hero_cards

    # Detect actions from chip changes, organized by street
    raw_actions: Dict[Street, List[HandAction]] = {
        s: [] for s in Street if s != Street.UNKNOWN
    }

    # Track which actions result in all-in (player's chips went to 0)
    all_in_actions: set = set()

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
                if prev_p and curr_p and pos < len(players):
                    chip_change = (prev_p.chips or 0) - (curr_p.chips or 0)
                    if chip_change > 0:
                        name = players[pos]
                        if chip_change > current_bet:
                            action_type = PlayerAction.BET if current_bet == 0 else PlayerAction.RAISE
                        else:
                            action_type = PlayerAction.CALL
                        if chip_change > current_bet:
                            current_bet = chip_change

                        action = HandAction(name, action_type, chip_change)
                        if current_street in raw_actions:
                            raw_actions[current_street].append(action)

                        # Track all-in (chips went to 0 after betting)
                        if curr_p.chips == 0:
                            all_in_actions.add(id(action))

        prev_state = state

    # Add blind posts as explicit actions at start of preflop
    # This allows preflop to be treated uniformly with other streets
    sb_name = players[sb_idx]
    bb_name = players[bb_idx]
    raw_actions[Street.PREFLOP].insert(0, HandAction(bb_name, PlayerAction.BET, big_blind))
    raw_actions[Street.PREFLOP].insert(0, HandAction(sb_name, PlayerAction.BET, small_blind))

    # players_in_hand: who hasn't folded (may include all-in players)
    players_in_hand = list(players)

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

    # Cumulative set of players who are all-in (added as we process actions)
    players_all_in: set = set()

    for street in streets:
        if not street_reached(street) or len(players_in_hand) <= 1:
            break

        actions_list = hand.actions[street]
        street_actions = raw_actions[street]

        # Contributions this street (starts empty, built from actions)
        contributions: Dict[str, int] = {}
        current_bet = 0

        # First player to act (SB for all streets - preflop has blind posts first)
        first_to_act_idx = sb_idx

        # Current position in the rotation (index into players list)
        current_pos = first_to_act_idx

        def get_next_active_player(start_pos: int, in_hand: List[str], all_in: set) -> tuple[int, str] | None:
            """Find next player who can act, starting from start_pos."""
            for i in range(len(players)):
                pos = (start_pos + i) % len(players)
                name = players[pos]
                if name in in_hand and name not in all_in:
                    return pos, name
            return None

        # Process each raw action in order
        for action in street_actions:
            if len(players_in_hand) <= 1:
                break

            actor = action.player_name

            # Walk from current position to actor, inferring checks/folds
            while True:
                result = get_next_active_player(current_pos, players_in_hand, players_all_in)
                if result is None:
                    break
                pos, name = result

                if name == actor:
                    # Found the actor - record their action
                    actions_list.append(action)
                    if action.amount:
                        contributions[name] = contributions.get(name, 0) + action.amount
                        if contributions[name] > current_bet:
                            current_bet = contributions[name]
                    # Mark as all-in if this action put them all-in
                    if id(action) in all_in_actions:
                        players_all_in.add(name)
                    current_pos = (pos + 1) % len(players)
                    break
                else:
                    # This player acted before the actor - infer check or fold
                    to_call = current_bet - contributions.get(name, 0)
                    if to_call > 0:
                        actions_list.append(HandAction(name, PlayerAction.FOLD, 0))
                        players_in_hand = [p for p in players_in_hand if p != name]
                    else:
                        actions_list.append(HandAction(name, PlayerAction.CHECK, 0))
                    current_pos = (pos + 1) % len(players)

        # After all raw actions, close out remaining players
        # - Players who owe chips must fold
        # - Players who don't owe (and haven't acted) must check
        if len(players_in_hand) > 1:
            for i in range(len(players)):
                pos = (current_pos + i) % len(players)
                name = players[pos]
                if name not in players_in_hand or name in players_all_in:
                    continue
                to_call = current_bet - contributions.get(name, 0)
                if to_call > 0:
                    actions_list.append(HandAction(name, PlayerAction.FOLD, 0))
                    players_in_hand = [p for p in players_in_hand if p != name]
                elif name not in contributions:
                    # Player hasn't acted yet and doesn't owe anything - they check
                    actions_list.append(HandAction(name, PlayerAction.CHECK, 0))

    # BB option: if action limped to BB preflop, BB gets option to check/raise
    # Count BB's voluntary actions (exclude the forced blind post which is first 2 actions)
    bb_in_hand = bb_name in players_in_hand
    bb_voluntary_actions = [
        a for a in hand.actions[Street.PREFLOP][2:]  # Skip SB and BB blind posts
        if a.player_name == bb_name
    ]
    if bb_in_hand and not bb_voluntary_actions and hand.flop_cards:
        hand.actions[Street.PREFLOP].append(HandAction(bb_name, PlayerAction.CHECK, 0))

    # Remove the synthetic blind posts from preflop actions
    del hand.actions[Street.PREFLOP][:2]

    # Determine outcome
    # Hero is always the last player in the list
    hero_player_name = players[-1]
    hand.reached_showdown = len(players_in_hand) > 1
    hand.opponents_folded = len(players_in_hand) == 1 and hero_player_name in players_in_hand
    hand.hero_folded = hero_player_name not in players_in_hand

    # Calculate uncalled bet - the portion of a bet that wasn't called
    all_actions = sum((hand.actions[s] for s in streets), [])

    last_bet_amount = 0
    last_bettor = None
    max_call_amount = 0

    for a in all_actions:
        if a.action in (PlayerAction.BET, PlayerAction.RAISE):
            last_bet_amount = a.amount or 0
            last_bettor = a.player_name
            max_call_amount = 0
        elif a.action == PlayerAction.CALL and last_bettor:
            max_call_amount = max(max_call_amount, a.amount or 0)

    uncalled_amount = 0
    if last_bettor and last_bet_amount > max_call_amount:
        uncalled_amount = last_bet_amount - max_call_amount
        # Add to the last street that has actions
        for street in reversed(streets):
            if hand.actions[street]:
                hand.actions[street].append(
                    HandAction(last_bettor, PlayerAction.UNCALLED_BET, uncalled_amount)
                )
                break

    # Set winner if everyone folded
    if len(players_in_hand) == 1:
        hand.winner = players_in_hand[0]
        hand.payout = hand.pot - uncalled_amount

    return hand


# Backwards compatibility alias
class HandReconstructor:
    """Deprecated: Use reconstruct_hand() function instead."""

    def __init__(
        self,
        player_names: Optional[List[str]] = None,
    ):
        self.player_names = player_names or []

    def reconstruct(
        self,
        states: List[GameState],
        button_pos: Optional[int] = None,
    ) -> Optional[HandHistory]:
        hero_cards: List[Card] = []
        for state in states:
            if state.hero_cards:
                hero_cards = state.hero_cards
                break
        return reconstruct_hand(states, self.player_names, button_pos, hero_cards)
