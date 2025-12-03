"""
Hand history reconstruction from GameState observations.

Shared data structures and reconstruction logic used by all exporters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.table_regions import PlayerPosition
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
    seat: int
    name: str
    chips: int
    position: PlayerPosition
    is_hero: bool = False


@dataclass
class HandHistory:
    """Complete history of a single poker hand."""
    hand_id: str
    table_name: str = "Governor of Poker"
    timestamp: datetime = field(default_factory=datetime.now)

    small_blind: int = 10
    big_blind: int = 20

    players: List[PlayerInfo] = field(default_factory=list)
    button_seat: int = 0
    sb_seat: int = 0
    bb_seat: int = 0

    hero_cards: List[Card] = field(default_factory=list)
    flop_cards: List[Card] = field(default_factory=list)
    turn_card: Optional[Card] = None
    river_card: Optional[Card] = None

    preflop_actions: List[HandAction] = field(default_factory=list)
    flop_actions: List[HandAction] = field(default_factory=list)
    turn_actions: List[HandAction] = field(default_factory=list)
    river_actions: List[HandAction] = field(default_factory=list)

    pot: int = 0
    hero_went_all_in: bool = False
    hero_folded: bool = False
    uncalled_bet: int = 0
    uncalled_bet_player: Optional[str] = None

    def get_sb_player(self) -> Optional[PlayerInfo]:
        for p in self.players:
            if p.seat == self.sb_seat:
                return p
        return None

    def get_bb_player(self) -> Optional[PlayerInfo]:
        for p in self.players:
            if p.seat == self.bb_seat:
                return p
        return None

    def get_hero(self) -> Optional[PlayerInfo]:
        for p in self.players:
            if p.is_hero:
                return p
        return None

    def get_opponents(self) -> List[PlayerInfo]:
        """Get non-hero players in post-flop action order (starting from SB)."""
        opponents = []
        for i in range(len(self.players)):
            idx = (self.sb_seat + i) % len(self.players)
            p = self.players[idx]
            if not p.is_hero:
                opponents.append(p)
        return opponents


def _calculate_blind_positions(button_seat: int, num_players: int) -> Tuple[int, int]:
    """Calculate SB and BB seats. Handles heads-up where button = SB."""
    if num_players == 2:
        return button_seat, (button_seat + 1) % 2
    elif num_players >= 3:
        return (button_seat + 1) % num_players, (button_seat + 2) % num_players
    return 0, 0


class HandReconstructor:
    """Reconstructs hand history from GameState observations."""

    SEAT_ORDER = [
        PlayerPosition.SEAT_1,
        PlayerPosition.SEAT_2,
        PlayerPosition.SEAT_3,
        PlayerPosition.SEAT_4,
        PlayerPosition.HERO,
    ]

    def __init__(
        self,
        hero_name: str = "hero",
        player_names: Optional[Dict[PlayerPosition, str]] = None,
    ):
        self.hero_name = hero_name
        self.player_names = player_names or {}

    def reconstruct(self, states: List[GameState], button_pos: int = 0) -> Optional[HandHistory]:
        if not states:
            return None

        initial, final = states[0], states[-1]

        # Infer blinds from initial pot (SB:BB = 1:2)
        initial_pot = initial.pot or 60
        small_blind = initial_pot // 3
        big_blind = small_blind * 2

        players, pos_to_player, orig_to_new = self._build_players(initial)
        if not players:
            return None

        new_button = orig_to_new.get(button_pos, 0)
        sb_seat, bb_seat = _calculate_blind_positions(new_button, len(players))

        hand = HandHistory(
            hand_id=str(abs(hash(str(initial.timestamp_ms))) % 100000000),
            small_blind=small_blind,
            big_blind=big_blind,
            players=players,
            button_seat=new_button,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            pot=final.pot or 0,
        )

        for state in states:
            if state.hero_cards:
                hand.hero_cards = state.hero_cards
                break

        self._process_states(hand, states, pos_to_player, big_blind)

        # Add BB check if missing (BB has option after SB completes, check has no pot change)
        bb_player = hand.get_bb_player()
        if bb_player and hand.flop_cards:
            bb_acted = any(a.player_name == bb_player.name for a in hand.preflop_actions)
            if not bb_acted:
                hand.preflop_actions.append(
                    HandAction(bb_player.name, PlayerAction.CHECK, 0)
                )

        # Add checks for streets with no actions (all players checked)
        self._add_missing_checks(hand)

        return hand

    def _add_missing_checks(self, hand: HandHistory):
        """Add check actions for streets where no actions were recorded."""
        # Get players in post-flop action order (SB first)
        active_players = []
        for i in range(len(hand.players)):
            idx = (hand.sb_seat + i) % len(hand.players)
            active_players.append(hand.players[idx])

        # For each post-flop street, if we have cards but no actions, add checks
        streets = [
            (hand.flop_cards, hand.flop_actions, hand.turn_card),
            (hand.turn_card, hand.turn_actions, hand.river_card),
            (hand.river_card, hand.river_actions, None),
        ]

        for street_cards, street_actions, next_street in streets:
            if street_cards and not street_actions and next_street:
                # Street exists, no actions recorded, and we went to next street = all checked
                for p in active_players:
                    street_actions.append(HandAction(p.name, PlayerAction.CHECK, 0))

    def _build_players(self, initial: GameState):
        players, orig_to_new, pos_to_player = [], {}, {}
        seat_idx = 0

        for orig_idx, pos in enumerate(self.SEAT_ORDER):
            if pos in initial.players:
                chips = initial.players[pos].chips
                if not chips or chips <= 0:
                    continue

                if pos == PlayerPosition.HERO:
                    name = self.hero_name
                elif pos in self.player_names:
                    name = self.player_names[pos]
                else:
                    name = pos.name
                player = PlayerInfo(seat_idx, name, chips, pos, pos == PlayerPosition.HERO)
                players.append(player)
                pos_to_player[pos] = player
                orig_to_new[orig_idx] = seat_idx
                seat_idx += 1

        return players, pos_to_player, orig_to_new

    def _process_states(self, hand: HandHistory, states: List[GameState], pos_to_player: dict, big_blind: int):
        current_street = Street.PREFLOP
        prev_state = states[0]
        current_bet = big_blind
        last_aggressor, last_bet = None, 0

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
                    if prev_p and curr_p and pos in pos_to_player:
                        chip_change = (prev_p.chips or 0) - (curr_p.chips or 0)
                        if chip_change > 0:
                            player = pos_to_player[pos]
                            action_type = PlayerAction.RAISE if chip_change > current_bet else PlayerAction.CALL
                            if chip_change > current_bet:
                                current_bet = chip_change

                            action = HandAction(player.name, action_type, chip_change)
                            self._add_action(hand, current_street, action)

                            last_aggressor, last_bet = player, chip_change

                            if player.is_hero and (curr_p.chips or 0) == 0:
                                hand.hero_went_all_in = True
                                hand.pot = prev_state.pot or 0
                                return

            prev_state = state

        hand.hero_folded = True
        hand.pot = states[-1].pot or 0
        if last_aggressor and last_bet > 0:
            hand.uncalled_bet = last_bet
            hand.uncalled_bet_player = last_aggressor.name
            hand.pot -= last_bet

    def _add_action(self, hand: HandHistory, street: Street, action: HandAction):
        {
            Street.PREFLOP: hand.preflop_actions,
            Street.FLOP: hand.flop_actions,
            Street.TURN: hand.turn_actions,
            Street.RIVER: hand.river_actions,
        }.get(street, []).append(action)
