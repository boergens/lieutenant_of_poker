"""
Hand history export for Governor of Poker.

Exports detected game states to standard hand history formats
for use with poker analysis tools.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, TextIO
import io

from lieutenant_of_poker.game_state import GameState, Street, PlayerState
from lieutenant_of_poker.table_regions import PlayerPosition
from lieutenant_of_poker.card_detector import Card
from lieutenant_of_poker.action_detector import PlayerAction


@dataclass
class HandAction:
    """A single action in a hand."""
    player_name: str
    action: PlayerAction
    amount: Optional[int] = None

    def __str__(self) -> str:
        if self.action == PlayerAction.FOLD:
            return f"{self.player_name}: folds"
        elif self.action == PlayerAction.CHECK:
            return f"{self.player_name}: checks"
        elif self.action == PlayerAction.CALL:
            if self.amount:
                return f"{self.player_name}: calls ${self.amount}"
            return f"{self.player_name}: calls"
        elif self.action == PlayerAction.RAISE:
            if self.amount:
                return f"{self.player_name}: raises to ${self.amount}"
            return f"{self.player_name}: raises"
        elif self.action == PlayerAction.BET:
            if self.amount:
                return f"{self.player_name}: bets ${self.amount}"
            return f"{self.player_name}: bets"
        elif self.action == PlayerAction.ALL_IN:
            if self.amount:
                return f"{self.player_name}: raises to ${self.amount} and is all-in"
            return f"{self.player_name}: is all-in"
        else:
            return f"{self.player_name}: unknown action"


@dataclass
class HandHistory:
    """Complete history of a single poker hand."""
    hand_id: str
    table_name: str = "Governor of Poker"
    timestamp: datetime = field(default_factory=datetime.now)

    # Stakes
    small_blind: int = 10
    big_blind: int = 20

    # Players and positions
    players: List[tuple] = field(default_factory=list)  # [(name, chips, position)]
    button_position: int = 0

    # Cards
    hero_cards: List[Card] = field(default_factory=list)
    flop_cards: List[Card] = field(default_factory=list)
    turn_card: Optional[Card] = None
    river_card: Optional[Card] = None

    # Actions by street
    preflop_actions: List[HandAction] = field(default_factory=list)
    flop_actions: List[HandAction] = field(default_factory=list)
    turn_actions: List[HandAction] = field(default_factory=list)
    river_actions: List[HandAction] = field(default_factory=list)

    # Results
    pot: int = 0
    winner: Optional[str] = None
    winnings: Optional[int] = None


class HandHistoryExporter:
    """Exports hand histories to various formats."""

    def __init__(self):
        """Initialize the exporter."""
        self._hand_counter = 0

    def export_pokerstars_format(self, hand: HandHistory) -> str:
        """
        Export hand history in PokerStars format.

        Args:
            hand: HandHistory object to export.

        Returns:
            String in PokerStars hand history format.
        """
        output = io.StringIO()
        self._write_pokerstars_format(hand, output)
        return output.getvalue()

    def _write_pokerstars_format(self, hand: HandHistory, f: TextIO) -> None:
        """Write hand history in PokerStars format."""
        # Header
        f.write(f"PokerStars Hand #{hand.hand_id}: Hold'em No Limit ")
        f.write(f"(${hand.small_blind}/${hand.big_blind})\n")
        f.write(f"Table '{hand.table_name}' 6-max Seat #{hand.button_position + 1} is the button\n")

        # Players
        for i, (name, chips, pos) in enumerate(hand.players):
            f.write(f"Seat {i + 1}: {name} (${chips} in chips)\n")

        # Blinds
        if len(hand.players) >= 2:
            sb_idx = (hand.button_position + 1) % len(hand.players)
            bb_idx = (hand.button_position + 2) % len(hand.players)
            f.write(f"{hand.players[sb_idx][0]}: posts small blind ${hand.small_blind}\n")
            f.write(f"{hand.players[bb_idx][0]}: posts big blind ${hand.big_blind}\n")

        # Hole cards
        f.write("*** HOLE CARDS ***\n")
        if hand.hero_cards:
            cards_str = " ".join(c.short_name for c in hand.hero_cards)
            f.write(f"Dealt to Hero [{cards_str}]\n")

        # Preflop actions
        for action in hand.preflop_actions:
            f.write(f"{action}\n")

        # Flop
        if hand.flop_cards:
            cards_str = " ".join(c.short_name for c in hand.flop_cards)
            f.write(f"*** FLOP *** [{cards_str}]\n")
            for action in hand.flop_actions:
                f.write(f"{action}\n")

        # Turn
        if hand.turn_card:
            flop_str = " ".join(c.short_name for c in hand.flop_cards)
            turn_str = hand.turn_card.short_name
            f.write(f"*** TURN *** [{flop_str}] [{turn_str}]\n")
            for action in hand.turn_actions:
                f.write(f"{action}\n")

        # River
        if hand.river_card:
            board_str = " ".join(c.short_name for c in hand.flop_cards)
            board_str += f" {hand.turn_card.short_name}"
            river_str = hand.river_card.short_name
            f.write(f"*** RIVER *** [{board_str}] [{river_str}]\n")
            for action in hand.river_actions:
                f.write(f"{action}\n")

        # Summary
        f.write("*** SUMMARY ***\n")
        f.write(f"Total pot ${hand.pot}\n")

        # Board
        board_cards = hand.flop_cards.copy()
        if hand.turn_card:
            board_cards.append(hand.turn_card)
        if hand.river_card:
            board_cards.append(hand.river_card)
        if board_cards:
            board_str = " ".join(c.short_name for c in board_cards)
            f.write(f"Board [{board_str}]\n")

        if hand.winner and hand.winnings:
            f.write(f"{hand.winner} collected ${hand.winnings} from pot\n")

    def create_hand_from_states(
        self,
        states: List[GameState],
        hero_name: str = "Hero"
    ) -> Optional[HandHistory]:
        """
        Create a HandHistory from a sequence of GameStates.

        Args:
            states: List of GameState objects from a single hand.
            hero_name: Name to use for the hero player.

        Returns:
            HandHistory object, or None if states are empty.
        """
        if not states:
            return None

        self._hand_counter += 1
        hand = HandHistory(hand_id=str(self._hand_counter))

        # Get initial state for player info
        initial = states[0]

        # Set hero cards from first state that has them
        for state in states:
            if state.hero_cards:
                hand.hero_cards = state.hero_cards
                break

        # Get community cards from final state
        final = states[-1]
        if final.community_cards:
            if len(final.community_cards) >= 3:
                hand.flop_cards = final.community_cards[:3]
            if len(final.community_cards) >= 4:
                hand.turn_card = final.community_cards[3]
            if len(final.community_cards) >= 5:
                hand.river_card = final.community_cards[4]

        # Get pot from final state
        if final.pot:
            hand.pot = final.pot

        # Build player list
        for pos, player in initial.players.items():
            name = player.name or f"Player_{pos.name}"
            chips = player.chips or 1000
            hand.players.append((name, chips, pos))

        return hand


def export_hand_to_pokerstars(hand: HandHistory) -> str:
    """
    Convenience function to export a hand in PokerStars format.

    Args:
        hand: HandHistory to export.

    Returns:
        PokerStars format string.
    """
    exporter = HandHistoryExporter()
    return exporter.export_pokerstars_format(hand)
