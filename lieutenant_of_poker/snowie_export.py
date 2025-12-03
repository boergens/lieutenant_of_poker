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
    # Generate hand_id from rng
    hand_id = str(rng.randint(10000000, 99999999))
    hand = HandReconstructor(hero_name, player_names).reconstruct(states, button_pos, hand_id=hand_id)
    if not hand:
        return ""
    return SnowieExporter(hero_name, rng).export(hand)


class SnowieExporter:
    """Exports HandHistory to Snowie/Freezeout format."""

    def __init__(self, hero_name: str = "hero", rng: RNG = random):
        self.hero_name = hero_name
        self.rng = rng

    def export(self, hand: HandHistory) -> str:
        output = io.StringIO()
        self._write(output, hand)
        return output.getvalue()

    def _write(self, f: TextIO, hand: HandHistory):
        # Hero is always the last player in the list
        hero = hand.players[-1] if hand.players else None
        hero_name = hero.name if hero else self.hero_name

        # Header
        f.write("GameStart\n")
        f.write("PokerClient: ExportFormat\n")
        f.write(f"Date: {hand.timestamp.strftime('%d/%m/%Y')}\n")
        f.write("TimeZone: GMT\n")
        f.write(f"Time: {hand.timestamp.strftime('%H:%M:%S')}\n")
        f.write(f"GameId:{hand.hand_id}\n")
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

        for p in hand.players:
            f.write(f"Seat {p.seat} {p.name} {p.chips}\n")

        sb, bb = hand.get_sb_player(), hand.get_bb_player()
        if sb and bb:
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
            self._write_hero_all_in(f, hand)
        elif hand.opponent_folded:
            self._write_opponent_folds(f, hand)
        elif hand.hero_folded:
            self._write_hero_fold(f, hand)
        elif hand.reached_showdown:
            self._write_showdown(f, hand)

        f.write("GameEnd\n\n")

    def _write_actions(self, f: TextIO, actions: List[HandAction]):
        for a in actions:
            if a.action == PlayerAction.FOLD:
                f.write(f"Move: {a.player_name} folds 0\n")
            elif a.action in (PlayerAction.CHECK, PlayerAction.CALL):
                f.write(f"Move: {a.player_name} call_check {a.amount or 0}\n")
            else:
                f.write(f"Move: {a.player_name} raise_bet {a.amount or 0}\n")

    def _write_hero_all_in(self, f: TextIO, hand: HandHistory):
        # Hero is the last player in the list
        hero = hand.players[-1] if hand.players else None
        hero_name = hero.name if hero else self.hero_name

        for p in hand.players:
            if p != hero:
                f.write(f"Move: {p.name} folds 0\n")

        # Return uncalled bet to hero
        if hand.uncalled_bet > 0:
            f.write(f"Move: {hero_name} uncalled_bet {hand.uncalled_bet}\n")

        f.write(f"Winner: {hero_name} {hand.pot:.2f}\n")

    def _write_opponent_folds(self, f: TextIO, hand: HandHistory):
        """Opponent folded to hero's bet - hero wins."""
        # Hero is the last player in the list
        hero = hand.players[-1] if hand.players else None
        hero_name = hero.name if hero else self.hero_name

        for p in hand.players:
            if p != hero:
                f.write(f"Move: {p.name} folds 0\n")

        # Return uncalled bet to hero
        if hand.uncalled_bet > 0:
            f.write(f"Move: {hero_name} uncalled_bet {hand.uncalled_bet}\n")

        f.write(f"Winner: {hero_name} {hand.pot:.2f}\n")

    def _write_hero_fold(self, f: TextIO, hand: HandHistory):
        f.write(f"Move: {self.hero_name} folds 0\n")

        if hand.uncalled_bet_player and hand.uncalled_bet > 0:
            f.write(f"Move: {hand.uncalled_bet_player} uncalled_bet {hand.uncalled_bet}\n")

        opponents = hand.get_opponents()
        sb, bb = hand.get_sb_player(), hand.get_bb_player()

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
                if sb and p.name == sb.name:
                    f.write(f"Move: {p.name} call_check {hand.small_blind}\n")
                elif bb and p.name == bb.name:
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

    def _write_showdown(self, f: TextIO, hand: HandHistory):
        """Write showdown when hand reaches river with no fold."""
        opponents = hand.get_opponents()

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
            f.write(f"Showdown: {self.hero_name} [{''.join(hero_cards)}]\n")

        # Write showdown for opponents (simulated cards)
        for name, cards in sim.opponent_hands.items():
            f.write(f"Showdown: {name} [{' '.join(cards)}]\n")

        f.write(f"Winner: {sim.winner} {hand.pot:.2f}\n")
