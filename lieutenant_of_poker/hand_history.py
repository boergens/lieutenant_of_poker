"""
Hand history reconstruction from GameState observations.

Shared data structures and reconstruction logic used by all exporters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.action_detector import PlayerAction


@dataclass
class ChipMovement:
    """Raw chip movement detected from frame-to-frame comparison."""
    player_name: str
    amount: int
    is_all_in: bool = False


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

    hero_cards: List[str] = field(default_factory=list)  # Card strings like "Ah"
    flop_cards: Optional[List[str]] = None
    turn_card: Optional[str] = None
    river_card: Optional[str] = None

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
    hero_cards: List[str],
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

    # Detect chip movements from frame-to-frame comparison, organized by street
    # This is a dumb pass - just record who put in how much, when
    chip_movements: Dict[Street, List[ChipMovement]] = {
        s: [] for s in Street if s != Street.UNKNOWN
    }

    current_street = Street.PREFLOP
    prev_state = states[0]

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

        pot_change = (state.pot or 0) - (prev_state.pot or 0)
        if pot_change > 0:
            for pos in state.players:
                prev_p, curr_p = prev_state.players.get(pos), state.players.get(pos)
                if prev_p and curr_p and pos < len(players):
                    chip_change = (prev_p.chips or 0) - (curr_p.chips or 0)
                    if chip_change > 0:
                        name = players[pos]
                        is_all_in = curr_p.chips == 0
                        movement = ChipMovement(name, chip_change, is_all_in)
                        if current_street in chip_movements:
                            chip_movements[current_street].append(movement)

        prev_state = state

    # Add blind posts as chip movements at start of preflop
    sb_name = players[sb_idx]
    bb_name = players[bb_idx]
    chip_movements[Street.PREFLOP].insert(0, ChipMovement(bb_name, big_blind))
    chip_movements[Street.PREFLOP].insert(0, ChipMovement(sb_name, small_blind))

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

    def get_next_active_player(start_pos: int, in_hand: List[str], all_in: set) -> tuple[int, str] | None:
        """Find next player who can act, starting from start_pos."""
        for i in range(len(players)):
            pos = (start_pos + i) % len(players)
            name = players[pos]
            if name in in_hand and name not in all_in:
                return pos, name
        return None

    for street in streets:
        if not street_reached(street) or len(players_in_hand) <= 1:
            break

        # If all but one player are all-in, no betting action needed on this street
        # The hand proceeds automatically to showdown
        active_players = [p for p in players_in_hand if p not in players_all_in]
        if len(active_players) <= 1:
            continue

        actions_list = hand.actions[street]
        street_movements = chip_movements[street]

        # Contributions this street (starts empty, built from movements)
        contributions: Dict[str, int] = {}
        current_bet = 0

        # First player to act (SB for all streets - preflop has blind posts first)
        first_to_act_idx = sb_idx

        # Current position in the rotation (index into players list)
        current_pos = first_to_act_idx

        # Process each chip movement and classify it based on betting context
        for movement in street_movements:
            if len(players_in_hand) <= 1:
                break

            actor = movement.player_name

            # Walk from current position to actor, inferring checks/folds
            while True:
                result = get_next_active_player(current_pos, players_in_hand, players_all_in)
                if result is None:
                    break
                pos, name = result

                if name == actor:
                    # Found the actor - classify and record their action
                    # Now we have full betting context to determine action type
                    new_total = contributions.get(name, 0) + movement.amount

                    if new_total > current_bet:
                        action_type = PlayerAction.BET if current_bet == 0 else PlayerAction.RAISE
                    else:
                        action_type = PlayerAction.CALL

                    action = HandAction(name, action_type, movement.amount)
                    actions_list.append(action)

                    contributions[name] = new_total
                    if new_total > current_bet:
                        current_bet = new_total

                    # Mark as all-in if this movement put them all-in
                    if movement.is_all_in:
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

        # After all movements, close out remaining players
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
    bb_voluntary_actions = [
        a for a in hand.actions[Street.PREFLOP][2:]  # Skip SB and BB blind posts
        if a.player_name == bb_name
    ]
    bb_folded_preflop = any(a.action == PlayerAction.FOLD for a in bb_voluntary_actions)
    if not bb_folded_preflop and not bb_voluntary_actions and hand.flop_cards:
        hand.actions[Street.PREFLOP].append(HandAction(bb_name, PlayerAction.CHECK, 0))

    # Calculate uncalled bet BEFORE removing synthetic blinds
    # Skip the blind posts (first 2 preflop actions) - blinds are mandatory,
    # not voluntary bets, so they don't count as "uncalled" if everyone folds.
    # But include their contribution amounts for proper totaling.
    all_actions = (
        hand.actions[Street.PREFLOP][2:] +  # Skip blind posts
        hand.actions[Street.FLOP] +
        hand.actions[Street.TURN] +
        hand.actions[Street.RIVER]
    )

    # Pre-seed contributions with blind amounts (they're not "bets" for uncalled purposes
    # but they do count toward total contributions)
    contributions: Dict[str, int] = {
        sb_name: small_blind,
        bb_name: big_blind,
    }
    last_bettor: Optional[str] = None
    last_bet_total = 0

    for a in all_actions:
        player = a.player_name
        amount = a.amount or 0

        if a.action in (PlayerAction.BET, PlayerAction.RAISE):
            contributions[player] = contributions.get(player, 0) + amount
            last_bettor = player
            last_bet_total = contributions[player]
        elif a.action == PlayerAction.CALL:
            contributions[player] = contributions.get(player, 0) + amount

    # Find max contribution from players other than last_bettor
    uncalled_amount = 0
    if last_bettor:
        max_other_contribution = 0
        for player, total in contributions.items():
            if player != last_bettor:
                max_other_contribution = max(max_other_contribution, total)

        if last_bet_total > max_other_contribution:
            uncalled_amount = last_bet_total - max_other_contribution
            # Add to the last street that has actions
            for street in reversed(streets):
                if hand.actions[street]:
                    hand.actions[street].append(
                        HandAction(last_bettor, PlayerAction.UNCALLED_BET, uncalled_amount)
                    )
                    break

    # Remove the synthetic blind posts from preflop actions
    del hand.actions[Street.PREFLOP][:2]

    # Determine outcome
    # Hero is always the last player in the list
    hero_player_name = players[-1]
    hand.reached_showdown = len(players_in_hand) > 1
    hand.opponents_folded = len(players_in_hand) == 1 and hero_player_name in players_in_hand
    hand.hero_folded = hero_player_name not in players_in_hand

    # Set winner if everyone folded
    if len(players_in_hand) == 1:
        hand.winner = players_in_hand[0]
        hand.payout = hand.pot - uncalled_amount

    return hand
