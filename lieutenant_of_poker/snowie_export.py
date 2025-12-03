"""Export hand history to PokerSnowie format."""

import io
import random
from typing import Dict, List, Optional, TextIO

from .hand_history import HandHistory, HandAction, HandReconstructor
from .game_state import GameState
from .action_detector import PlayerAction
from .game_simulator import simulate_hand_completion, RNG


def export_snowie(
    states: List[GameState],
    hero_name: str = "hero",
    button_pos: Optional[int] = None,
    player_names: Optional[Dict[int, str]] = None,
    rng: Optional[RNG] = None,
) -> str:
    """Export GameStates to Snowie format. Auto-detects button if not specified."""
    if rng is None:
        rng = random
    hand = HandReconstructor(hero_name, player_names).reconstruct(states, button_pos)
    if not hand:
        return ""
    hand_id = str(rng.randint(10000000, 99999999))
    return SnowieExporter(hero_name, rng).export(hand, hand_id)


class SnowieExporter:
    """Exports HandHistory to Snowie/Freezeout format."""

    def __init__(self, hero_name: str = "hero", rng: RNG = random):
        self.hero_name = hero_name
        self.rng = rng

    def export(self, hand: HandHistory, hand_id: str) -> str:
        output = io.StringIO()
        self._write(output, hand, hand_id)
        return output.getvalue()

    def _write(self, f: TextIO, hand: HandHistory, hand_id: str):
        # Hero is always the last player in the list
        hero = hand.players[-1]
        hero_name = hero.name
        opponents = hand.players[:-1]
        sb = hand.players[hand.sb_seat]
        bb = hand.players[hand.bb_seat]

        # Header
        f.write("GameStart\n")
        f.write("PokerClient: ExportFormat\n")
        f.write(f"Date: {hand.timestamp.strftime('%d/%m/%Y')}\n")
        f.write("TimeZone: GMT\n")
        f.write(f"Time: {hand.timestamp.strftime('%H:%M:%S')}\n")
        f.write(f"GameId:{hand_id}\n")
        f.write("GameType:NoLimit\n")
        f.write("GameCurrency: $\n")
        f.write(f"SmallBlindStake: {hand.small_blind}\n")
        f.write(f"BigBlindStake: {hand.big_blind}\n")
        f.write("AnteStake: 0\n")
        f.write(f"TableName: {hand.table_name}\n")
        f.write(f"Max number of players: {len(hand.players)}\n")
        f.write(f"MyPlayerName: {hero_name}\n")
        # Heads-up: dealer position is inverted in Snowie format
        if len(hand.players) == 2:
            dealer_pos = 1 - hand.button_seat
        else:
            dealer_pos = hand.button_seat
        f.write(f"DealerPosition: {dealer_pos}\n")

        for i, p in enumerate(hand.players):
            f.write(f"Seat {i} {p.name} {p.chips}\n")

        f.write(f"SmallBlind: {sb.name} {hand.small_blind}\n")
        f.write(f"BigBlind: {bb.name} {hand.big_blind}\n")

        if hand.hero_cards:
            f.write(f"Dealt Cards: [{''.join(c.short_name for c in hand.hero_cards)}]\n")

        # Preflop
        self._write_actions(f, hand.preflop_actions)

        # Post-flop streets
        if hand.flop_cards:
            f.write(f"FLOP Community Cards:[{' '.join(c.short_name for c in hand.flop_cards)}]\n")
            self._write_actions(f, hand.flop_actions)

        if hand.turn_card:
            board = [c.short_name for c in hand.flop_cards] + [hand.turn_card.short_name]
            f.write(f"TURN Community Cards:[{' '.join(board)}]\n")
            self._write_actions(f, hand.turn_actions)

        if hand.river_card:
            board = [c.short_name for c in hand.flop_cards]
            board += [hand.turn_card.short_name, hand.river_card.short_name]
            f.write(f"RIVER Community Cards:[{' '.join(board)}]\n")
            self._write_actions(f, hand.river_actions)

        # Ending
        if hand.hero_went_all_in:
            self._write_hero_all_in(f, hand, hero_name, opponents)
        elif hand.opponents_folded:
            self._write_opponents_fold(f, hand, hero_name, opponents)
        elif hand.hero_folded:
            self._write_hero_fold(f, hand, hero_name, opponents, sb, bb)
        elif hand.reached_showdown:
            self._write_showdown(f, hand, hero_name, opponents)

        f.write("GameEnd\n\n")

    def _write_actions(self, f: TextIO, actions: List[HandAction]):
        for a in actions:
            if a.action == PlayerAction.FOLD:
                f.write(f"Move: {a.player_name} folds 0\n")
            elif a.action in (PlayerAction.CHECK, PlayerAction.CALL):
                f.write(f"Move: {a.player_name} call_check {a.amount or 0}\n")
            else:
                f.write(f"Move: {a.player_name} raise_bet {a.amount or 0}\n")

    def _write_hero_all_in(self, f: TextIO, hand: HandHistory, hero_name: str, opponents):
        for p in opponents:
            f.write(f"Move: {p.name} folds 0\n")
        f.write(f"Winner: {hero_name} {hand.pot:.2f}\n")

    def _write_opponents_fold(self, f: TextIO, hand: HandHistory, hero_name: str, opponents):
        """Opponents folded to hero's bet - hero wins."""
        for p in opponents:
            f.write(f"Move: {p.name} folds 0\n")
        f.write(f"Winner: {hero_name} {hand.pot:.2f}\n")

    def _write_hero_fold(self, f: TextIO, hand: HandHistory, hero_name: str, opponents, sb, bb):
        f.write(f"Move: {hero_name} folds 0\n")

        has_flop = bool(hand.flop_cards)
        has_turn = hand.turn_card is not None
        has_river = hand.river_card is not None

        community = []
        if hand.flop_cards:
            community = [c.short_name for c in hand.flop_cards]
        if hand.turn_card:
            community.append(hand.turn_card.short_name)
        if hand.river_card:
            community.append(hand.river_card.short_name)

        # Simulate remaining preflop
        if not has_flop:
            for p in opponents:
                if p.name == sb.name:
                    f.write(f"Move: {p.name} call_check {hand.small_blind}\n")
                elif p.name == bb.name:
                    f.write(f"Move: {p.name} call_check 0\n")
                else:
                    f.write(f"Move: {p.name} call_check {hand.big_blind}\n")

        hero_cards = [c.short_name for c in hand.hero_cards] if hand.hero_cards else []
        sim = simulate_hand_completion(hero_cards, community, [p.name for p in opponents], hand.pot, self.rng)

        if not has_flop:
            f.write(f"FLOP Community Cards:[{' '.join(sim.flop)}]\n")
            for p in opponents:
                f.write(f"Move: {p.name} call_check 0\n")

        if not has_turn:
            f.write(f"TURN Community Cards:[{' '.join(sim.flop + [sim.turn])}]\n")
            for p in opponents:
                f.write(f"Move: {p.name} call_check 0\n")

        if not has_river:
            f.write(f"RIVER Community Cards:[{' '.join(sim.flop + [sim.turn, sim.river])}]\n")
            for p in opponents:
                f.write(f"Move: {p.name} call_check 0\n")

        for name, cards in sim.opponent_hands.items():
            f.write(f"Showdown: {name} [{' '.join(cards)}]\n")

        f.write(f"Winner: {sim.winner} {hand.pot:.2f}\n")

    def _write_showdown(self, f: TextIO, hand: HandHistory, hero_name: str, opponents):
        """Write showdown when hand reaches river with no fold."""
        community = []
        if hand.flop_cards:
            community = [c.short_name for c in hand.flop_cards]
        if hand.turn_card:
            community.append(hand.turn_card.short_name)
        if hand.river_card:
            community.append(hand.river_card.short_name)

        hero_cards = [c.short_name for c in hand.hero_cards] if hand.hero_cards else []
        sim = simulate_hand_completion(hero_cards, community, [p.name for p in opponents], hand.pot, self.rng)

        # Write showdown for hero
        if hand.hero_cards:
            f.write(f"Showdown: {hero_name} [{' '.join(hero_cards)}]\n")

        # Write showdown for opponents (simulated cards)
        for name, cards in sim.opponent_hands.items():
            f.write(f"Showdown: {name} [{' '.join(cards)}]\n")

        f.write(f"Winner: {sim.winner} {hand.pot:.2f}\n")
