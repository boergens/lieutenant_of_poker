"""Export hand history to PokerStars format."""

import io
import random
from typing import Dict, List, Optional

from .hand_history import HandHistory, HandAction, HandReconstructor, PlayerInfo
from .game_state import GameState
from .action_detector import PlayerAction
from .game_simulator import RNG


def export_pokerstars(
    states: List[GameState],
    hero_name: str = "hero",
    button_pos: int = 0,
    player_names: Optional[Dict[int, str]] = None,
    rng: Optional[RNG] = None,
) -> str:
    """Export GameStates to PokerStars format."""
    if rng is None:
        rng = random
    hand_id = str(rng.randint(10000000, 99999999))
    hand = HandReconstructor(hero_name, player_names).reconstruct(states, button_pos, hand_id=hand_id)
    if not hand:
        return ""
    return PokerStarsExporter(hero_name).export(hand)


class PokerStarsExporter:
    """Exports HandHistory to PokerStars format."""

    def __init__(self, hero_name: str = "hero"):
        self.hero_name = hero_name

    def export(self, hand: HandHistory) -> str:
        output = io.StringIO()
        f = output

        # Header
        f.write(f"PokerStars Hand #{hand.hand_id}: Hold'em No Limit ")
        f.write(f"(${hand.small_blind}/${hand.big_blind})\n")
        f.write(f"Table '{hand.table_name}' 6-max Seat #{hand.button_seat + 1} is the button\n")

        # Players
        for p in hand.players:
            f.write(f"Seat {p.seat + 1}: {p.name} (${p.chips} in chips)\n")

        # Blinds
        sb, bb = hand.get_sb_player(), hand.get_bb_player()
        if sb and bb:
            f.write(f"{sb.name}: posts small blind ${hand.small_blind}\n")
            f.write(f"{bb.name}: posts big blind ${hand.big_blind}\n")

        # Hole cards
        f.write("*** HOLE CARDS ***\n")
        if hand.hero_cards:
            cards = " ".join(c.short_name for c in hand.hero_cards)
            f.write(f"Dealt to {self.hero_name} [{cards}]\n")

        # Preflop
        for a in hand.preflop_actions:
            f.write(f"{self._format_action(a)}\n")

        # Flop
        if hand.flop_cards:
            cards = " ".join(c.short_name for c in hand.flop_cards)
            f.write(f"*** FLOP *** [{cards}]\n")
            for a in hand.flop_actions:
                f.write(f"{self._format_action(a)}\n")

        # Turn
        if hand.turn_card:
            flop = " ".join(c.short_name for c in hand.flop_cards)
            f.write(f"*** TURN *** [{flop}] [{hand.turn_card.short_name}]\n")
            for a in hand.turn_actions:
                f.write(f"{self._format_action(a)}\n")

        # River
        if hand.river_card:
            board = " ".join(c.short_name for c in hand.flop_cards)
            board += f" {hand.turn_card.short_name}"
            f.write(f"*** RIVER *** [{board}] [{hand.river_card.short_name}]\n")
            for a in hand.river_actions:
                f.write(f"{self._format_action(a)}\n")

        # Summary
        f.write("*** SUMMARY ***\n")
        f.write(f"Total pot ${hand.pot}\n")

        board_cards = list(hand.flop_cards) if hand.flop_cards else []
        if hand.turn_card:
            board_cards.append(hand.turn_card)
        if hand.river_card:
            board_cards.append(hand.river_card)
        if board_cards:
            board = " ".join(c.short_name for c in board_cards)
            f.write(f"Board [{board}]\n")

        return output.getvalue()

    def _format_action(self, action: HandAction) -> str:
        name = action.player_name
        if action.action == PlayerAction.FOLD:
            return f"{name}: folds"
        elif action.action == PlayerAction.CHECK:
            return f"{name}: checks"
        elif action.action == PlayerAction.CALL:
            return f"{name}: calls ${action.amount}" if action.amount else f"{name}: calls"
        elif action.action == PlayerAction.RAISE:
            return f"{name}: raises to ${action.amount}" if action.amount else f"{name}: raises"
        elif action.action == PlayerAction.BET:
            return f"{name}: bets ${action.amount}" if action.amount else f"{name}: bets"
        elif action.action == PlayerAction.ALL_IN:
            return f"{name}: raises to ${action.amount} and is all-in" if action.amount else f"{name}: is all-in"
        return f"{name}: unknown action"
