"""Export hand history as a simple chronological action log."""

from typing import List, Optional

from .hand_history import HandHistory, reconstruct_hand
from .game_state import GameState, Street
from .action_detector import PlayerAction


def export_action_log(
    states: List[GameState],
    button_pos: Optional[int] = None,
    player_names: Optional[List[str]] = None,
) -> str:
    """Export GameStates as a simple chronological action log."""
    hero_cards = []
    for state in states:
        if state.hero_cards:
            hero_cards = state.hero_cards
            break
    hand = reconstruct_hand(states, player_names or [], button_pos, hero_cards)
    if not hand:
        return "No hand data."
    return ActionLogExporter().export(hand)


class ActionLogExporter:
    """Exports HandHistory as a simple chronological list of actions."""

    def export(self, hand: HandHistory) -> str:
        lines = []

        # Blinds
        sb = hand.players[hand.sb_seat]
        bb = hand.players[hand.bb_seat]
        lines.append(f"{sb.name} posts small blind ${hand.small_blind}")
        lines.append(f"{bb.name} posts big blind ${hand.big_blind}")

        # Hole cards
        hero = hand.players[-1]
        if hand.hero_cards:
            cards = " ".join(c.short_name for c in hand.hero_cards)
            lines.append(f"Dealer deals hole cards to {hero.name}: [{cards}]")

        # Preflop actions
        for a in hand.actions[Street.PREFLOP]:
            lines.append(self._format_action(a))

        # Flop
        if hand.flop_cards:
            cards = " ".join(c.short_name for c in hand.flop_cards)
            lines.append(f"Dealer reveals flop: {cards}")
            for a in hand.actions[Street.FLOP]:
                lines.append(self._format_action(a))

        # Turn
        if hand.turn_card:
            lines.append(f"Dealer reveals turn: {hand.turn_card.short_name}")
            for a in hand.actions[Street.TURN]:
                lines.append(self._format_action(a))

        # River
        if hand.river_card:
            lines.append(f"Dealer reveals river: {hand.river_card.short_name}")
            for a in hand.actions[Street.RIVER]:
                lines.append(self._format_action(a))

        return "\n".join(lines)

    def _format_action(self, a) -> str:
        if a.action == PlayerAction.FOLD:
            return f"{a.player_name} folds"
        elif a.action == PlayerAction.CHECK:
            return f"{a.player_name} checks"
        elif a.action == PlayerAction.CALL:
            return f"{a.player_name} calls ${a.amount}"
        elif a.action == PlayerAction.RAISE:
            return f"{a.player_name} raises to ${a.amount}"
        elif a.action == PlayerAction.BET:
            return f"{a.player_name} bets ${a.amount}"
        elif a.action == PlayerAction.ALL_IN:
            return f"{a.player_name} goes all-in ${a.amount}"
        elif a.action == PlayerAction.UNCALLED_BET:
            return f"Uncalled bet (${a.amount}) returned to {a.player_name}"
        else:
            return f"{a.player_name} acts"
