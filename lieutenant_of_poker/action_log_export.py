"""Export hand history as a simple chronological action log."""

import random
from typing import Dict, List, Optional

from .hand_history import HandHistory, HandReconstructor
from .game_state import GameState
from .action_detector import PlayerAction
from .game_simulator import RNG


def export_action_log(
    states: List[GameState],
    hero_name: str = "hero",
    button_pos: Optional[int] = None,
    player_names: Optional[Dict[int, str]] = None,
    rng: Optional[RNG] = None,
) -> str:
    """Export GameStates as a simple chronological action log."""
    if rng is None:
        rng = random
    hand_id = str(rng.randint(10000000, 99999999))
    hand = HandReconstructor(hero_name, player_names).reconstruct(states, button_pos, hand_id=hand_id)
    if not hand:
        return "No hand data."
    return ActionLogExporter().export(hand)


class ActionLogExporter:
    """Exports HandHistory as a simple chronological list of actions."""

    def export(self, hand: HandHistory) -> str:
        lines = []

        # Blinds
        sb = hand.get_sb_player()
        bb = hand.get_bb_player()
        if sb:
            lines.append(f"{sb.name} posts small blind ${hand.small_blind}")
        if bb:
            lines.append(f"{bb.name} posts big blind ${hand.big_blind}")

        # Hole cards
        hero = hand.get_hero()
        if hand.hero_cards and hero:
            cards = " ".join(c.short_name for c in hand.hero_cards)
            lines.append(f"Dealer deals hole cards to {hero.name}: [{cards}]")

        # Preflop actions
        for a in hand.preflop_actions:
            lines.append(self._format_action(a))

        # Flop
        if hand.flop_cards:
            cards = " ".join(c.short_name for c in hand.flop_cards)
            lines.append(f"Dealer reveals flop: {cards}")
            for a in hand.flop_actions:
                lines.append(self._format_action(a))

        # Turn
        if hand.turn_card:
            lines.append(f"Dealer reveals turn: {hand.turn_card.short_name}")
            for a in hand.turn_actions:
                lines.append(self._format_action(a))

        # River
        if hand.river_card:
            lines.append(f"Dealer reveals river: {hand.river_card.short_name}")
            for a in hand.river_actions:
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
        else:
            return f"{a.player_name} acts"
