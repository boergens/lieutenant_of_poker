"""Export hand history in human-readable format."""

import io
from typing import List, Optional

from .export import HandHistory, reconstruct_hand
from .game_state import GameState, Street
from .action_detector import PlayerAction


def export_human(
    states: List[GameState],
    button_pos: Optional[int] = None,
    player_names: Optional[List[str]] = None,
) -> str:
    """Export GameStates to human-readable format. Auto-detects button if not specified."""
    hero_cards = []
    for state in states:
        if state.hero_cards:
            hero_cards = state.hero_cards
            break
    hand = reconstruct_hand(states, player_names or [], button_pos, hero_cards)
    if not hand:
        return "No hand data."
    return HumanExporter().export(hand)


class HumanExporter:
    """Exports HandHistory in human-readable story format."""

    def export(self, hand: HandHistory) -> str:
        output = io.StringIO()
        f = output

        # Header
        f.write(f"Stakes: ${hand.small_blind}/${hand.big_blind}\n")
        f.write(f"Table: {hand.table_name}\n\n")

        # Players
        f.write("Players:\n")
        for i, p in enumerate(hand.players):
            role = ""
            if i == hand.button_seat:
                role = " (BTN)"
            if i == hand.sb_seat:
                role = " (SB)"
            if i == hand.bb_seat:
                role = " (BB)"
            hero_mark = " *" if i == len(hand.players) - 1 else ""
            f.write(f"  Seat {i + 1}: {p.name} - ${p.chips}{role}{hero_mark}\n")

        # Blinds
        sb = hand.players[hand.sb_seat]
        bb = hand.players[hand.bb_seat]
        f.write(f"\n{sb.name} posts small blind ${hand.small_blind}\n")
        f.write(f"{bb.name} posts big blind ${hand.big_blind}\n")

        # Hero cards
        hero = hand.players[-1]
        if hand.hero_cards:
            cards = " ".join(c.short_name for c in hand.hero_cards)
            f.write(f"\n{hero.name} is dealt: [{cards}]\n")

        # Preflop
        f.write("\n--- PREFLOP ---\n")
        self._write_actions(f, hand.actions[Street.PREFLOP])

        # Flop
        if hand.flop_cards:
            cards = " ".join(c.short_name for c in hand.flop_cards)
            f.write(f"\n--- FLOP [{cards}] ---\n")
            self._write_actions(f, hand.actions[Street.FLOP])

        # Turn
        if hand.turn_card:
            flop = " ".join(c.short_name for c in hand.flop_cards)
            f.write(f"\n--- TURN [{flop} {hand.turn_card.short_name}] ---\n")
            self._write_actions(f, hand.actions[Street.TURN])

        # River
        if hand.river_card:
            board = " ".join(c.short_name for c in hand.flop_cards)
            board += f" {hand.turn_card.short_name} {hand.river_card.short_name}"
            f.write(f"\n--- RIVER [{board}] ---\n")
            self._write_actions(f, hand.actions[Street.RIVER])

        # Ending
        f.write("\n--- RESULT ---\n")
        if hand.reached_showdown:
            f.write("Went to showdown\n")
        elif hand.hero_folded:
            f.write("Hero folded\n")
        elif hand.opponents_folded:
            f.write("Opponents folded\n")
        f.write(f"Final pot: ${hand.pot}\n")

        return output.getvalue()

    def _write_actions(self, f, actions):
        if not actions:
            f.write("  (no actions)\n")
            return

        for a in actions:
            if a.action == PlayerAction.FOLD:
                f.write(f"  {a.player_name} folds\n")
            elif a.action == PlayerAction.CHECK:
                f.write(f"  {a.player_name} checks\n")
            elif a.action == PlayerAction.CALL:
                f.write(f"  {a.player_name} calls ${a.amount}\n")
            elif a.action == PlayerAction.RAISE:
                f.write(f"  {a.player_name} raises to ${a.amount}\n")
            elif a.action == PlayerAction.BET:
                f.write(f"  {a.player_name} bets ${a.amount}\n")
            elif a.action == PlayerAction.ALL_IN:
                f.write(f"  {a.player_name} is all-in for ${a.amount}\n")
            else:
                f.write(f"  {a.player_name} unknown action\n")
