"""
Video analyzer for Governor of Poker.

Assembles per-frame GameState objects into a complete game log including:
- Hand boundaries
- Player actions per street
- Winners and pot sizes
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from .game_state import GameState, Street
from .table_regions import PlayerPosition
from .action_detector import PlayerAction
from .card_detector import Card


@dataclass
class ActionEvent:
    """A recorded player action."""
    timestamp_ms: float
    position: PlayerPosition
    action: PlayerAction
    amount: Optional[int] = None

    def __str__(self) -> str:
        name = self.position.name
        if self.amount:
            return f"{name}: {self.action.name} ${self.amount:,}"
        return f"{name}: {self.action.name}"


@dataclass
class HandRecord:
    """Complete record of a single poker hand."""
    hand_number: int
    start_time_ms: float
    end_time_ms: Optional[float] = None

    # Cards
    hero_cards: List[Card] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)

    # Actions by street
    preflop_actions: List[ActionEvent] = field(default_factory=list)
    flop_actions: List[ActionEvent] = field(default_factory=list)
    turn_actions: List[ActionEvent] = field(default_factory=list)
    river_actions: List[ActionEvent] = field(default_factory=list)

    # Results
    final_pot: Optional[int] = None
    winner: Optional[PlayerPosition] = None

    # Player chip counts at start
    starting_chips: Dict[PlayerPosition, int] = field(default_factory=dict)

    # Blinds (deduced from action order and pot)
    small_blind_pos: Optional[PlayerPosition] = None
    big_blind_pos: Optional[PlayerPosition] = None
    small_blind_amount: Optional[int] = None
    big_blind_amount: Optional[int] = None
    initial_pot: Optional[int] = None

    def add_action(self, street: Street, action: ActionEvent) -> None:
        """Add an action to the appropriate street."""
        if street == Street.PREFLOP:
            self.preflop_actions.append(action)
        elif street == Street.FLOP:
            self.flop_actions.append(action)
        elif street == Street.TURN:
            self.turn_actions.append(action)
        elif street == Street.RIVER:
            self.river_actions.append(action)

    def __str__(self) -> str:
        lines = [f"=== Hand #{self.hand_number} ==="]

        # Show blinds info
        if self.small_blind_pos and self.big_blind_pos:
            sb_amt = f"${self.small_blind_amount:,}" if self.small_blind_amount else "SB"
            bb_amt = f"${self.big_blind_amount:,}" if self.big_blind_amount else "BB"
            lines.append(f"Blinds: {self.small_blind_pos.name} ({sb_amt}) / {self.big_blind_pos.name} ({bb_amt})")

        if self.hero_cards:
            cards = " ".join(str(c) for c in self.hero_cards)
            lines.append(f"Hero: {cards}")

        if self.community_cards:
            cards = " ".join(str(c) for c in self.community_cards)
            lines.append(f"Board: {cards}")

        for street_name, actions in [
            ("PREFLOP", self.preflop_actions),
            ("FLOP", self.flop_actions),
            ("TURN", self.turn_actions),
            ("RIVER", self.river_actions),
        ]:
            if actions:
                lines.append(f"--- {street_name} ---")
                for action in actions:
                    lines.append(f"  {action}")

        if self.final_pot:
            lines.append(f"Pot: ${self.final_pot:,}")

        if self.winner:
            lines.append(f"Winner: {self.winner.name}")

        return "\n".join(lines)


@dataclass
class GameLog:
    """Complete log of a poker session."""
    source: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    hands: List[HandRecord] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Game Log: {self.source}",
            f"Hands played: {len(self.hands)}",
            "",
        ]
        for hand in self.hands:
            lines.append(str(hand))
            lines.append("")
        return "\n".join(lines)


def _get_position_order() -> List[PlayerPosition]:
    """Get positions in clockwise order around the table."""
    return [
        PlayerPosition.SEAT_1,
        PlayerPosition.SEAT_2,
        PlayerPosition.SEAT_3,
        PlayerPosition.SEAT_4,
        PlayerPosition.HERO,
    ]


def _position_before(pos: PlayerPosition) -> PlayerPosition:
    """Get the position before (to the right of) the given position."""
    order = _get_position_order()
    idx = order.index(pos)
    return order[(idx - 1) % len(order)]


def _deduce_blinds(hand: HandRecord) -> None:
    """Deduce blind positions from first preflop action."""
    if not hand.preflop_actions:
        return

    # First preflop actor is UTG (position after BB)
    first_actor = hand.preflop_actions[0].position
    hand.big_blind_pos = _position_before(first_actor)
    hand.small_blind_pos = _position_before(hand.big_blind_pos)

    # Try to deduce blind amounts from initial pot
    # Initial pot = SB + BB, and typically BB = 2 * SB
    if hand.initial_pot:
        # Assuming standard 1:2 ratio
        hand.small_blind_amount = hand.initial_pot // 3
        hand.big_blind_amount = hand.initial_pot - hand.small_blind_amount


def _deduce_action_type(
    chip_decrease: int,
    current_bet_to_match: int,
    is_first_bet_in_round: bool,
    remaining_chips: int,
) -> PlayerAction:
    """
    Deduce action type from chip decrease.

    Args:
        chip_decrease: How much the player's chips decreased.
        current_bet_to_match: The current bet amount to call.
        is_first_bet_in_round: True if no one has bet yet this round.
        remaining_chips: Player's chips after the action.

    Returns:
        The deduced PlayerAction.
    """
    if remaining_chips == 0:
        return PlayerAction.ALL_IN

    if is_first_bet_in_round:
        return PlayerAction.BET

    if chip_decrease > current_bet_to_match:
        return PlayerAction.RAISE

    return PlayerAction.CALL


def assemble_game_log(states: List[GameState], source: str = "") -> GameLog:
    """
    Assemble a game log from a sequence of GameState objects.

    Deduces player actions from chip changes between frames.

    Args:
        states: List of GameState objects in chronological order.
        source: Optional source identifier (e.g., video filename).

    Returns:
        GameLog with detected hands and actions.
    """
    log = GameLog(source=source)

    if not states:
        return log

    # State tracking
    current_hand: Optional[HandRecord] = None
    hand_number = 0
    last_community_count = 0
    first_pot_seen: Optional[int] = None

    # Chip tracking for action deduction
    last_chips: Dict[PlayerPosition, Optional[int]] = {}
    last_pot: Optional[int] = None
    current_street_bet: int = 0  # Current bet amount to call this street
    players_acted_this_street: set = set()

    for state in states:
        current_ms = state.timestamp_ms or 0

        # Detect new hand (community cards reset)
        community_count = len(state.community_cards)
        if community_count == 0 and last_community_count > 0:
            # Hand ended, deduce blinds and save it
            if current_hand is not None:
                current_hand.end_time_ms = current_ms
                _deduce_blinds(current_hand)
                log.hands.append(current_hand)

            hand_number += 1
            current_hand = HandRecord(
                hand_number=hand_number,
                start_time_ms=current_ms,
            )
            first_pot_seen = None
            last_chips.clear()
            last_pot = None
            current_street_bet = 0
            players_acted_this_street.clear()

            # Record starting chips
            for pos, player in state.players.items():
                if player.chips:
                    current_hand.starting_chips[pos] = player.chips
                    last_chips[pos] = player.chips

        # Start first hand if needed
        if current_hand is None:
            hand_number += 1
            current_hand = HandRecord(
                hand_number=hand_number,
                start_time_ms=current_ms,
            )
            for pos, player in state.players.items():
                if player.chips:
                    current_hand.starting_chips[pos] = player.chips
                    last_chips[pos] = player.chips

        # Detect street change - reset betting round tracking
        if community_count != last_community_count and community_count > 0:
            current_street_bet = 0
            players_acted_this_street.clear()

        # Update hero cards
        if state.hero_cards and not current_hand.hero_cards:
            current_hand.hero_cards = state.hero_cards.copy()

        # Update community cards
        if len(state.community_cards) > len(current_hand.community_cards):
            current_hand.community_cards = state.community_cards.copy()

        # Track initial pot (first pot seen, before actions modify it)
        if state.pot and first_pot_seen is None:
            first_pot_seen = state.pot
            current_hand.initial_pot = first_pot_seen

        # Update final pot
        if state.pot:
            current_hand.final_pot = state.pot

        # Detect actions from chip changes
        for pos, player in state.players.items():
            current_chips = player.chips
            prev_chips = last_chips.get(pos)

            if current_chips is not None and prev_chips is not None:
                chip_decrease = prev_chips - current_chips

                # Only record if chips decreased (player made a bet)
                if chip_decrease > 0:
                    is_first_bet = current_street_bet == 0
                    action_type = _deduce_action_type(
                        chip_decrease=chip_decrease,
                        current_bet_to_match=current_street_bet,
                        is_first_bet_in_round=is_first_bet,
                        remaining_chips=current_chips,
                    )

                    event = ActionEvent(
                        timestamp_ms=current_ms,
                        position=pos,
                        action=action_type,
                        amount=chip_decrease,
                    )
                    current_hand.add_action(state.street, event)

                    # Update betting state
                    if chip_decrease > current_street_bet:
                        current_street_bet = chip_decrease
                    players_acted_this_street.add(pos)

            # Update chip tracking
            if current_chips is not None:
                last_chips[pos] = current_chips

        last_pot = state.pot
        last_community_count = community_count

    # Save final hand
    if current_hand is not None:
        if states:
            current_hand.end_time_ms = states[-1].timestamp_ms
        _deduce_blinds(current_hand)
        log.hands.append(current_hand)

    return log
