"""Export hand history to PokerSnowie format."""

import io
from typing import List, Optional, TextIO

from .hand_history import HandHistory, HandAction, reconstruct_hand
from .game_state import GameState, Street
from .action_detector import PlayerAction
from .game_simulator import ShowdownConfig, pick_winner


def export_snowie(
    states: List[GameState],
    button_pos: Optional[int] = None,
    player_names: Optional[List[str]] = None,
    showdown: Optional[ShowdownConfig] = None,
    hand_id: Optional[str] = None,
) -> str:
    """Export GameStates to Snowie format.

    Args:
        states: Game states to export
        button_pos: Button position (auto-detected if not specified)
        player_names: List of player names
        showdown: Showdown configuration with opponent cards for deterministic output.
        hand_id: Optional hand ID (random 8-digit number if not provided)
    """
    hero_cards = []
    for state in states:
        if state.hero_cards:
            hero_cards = state.hero_cards
            break
    hand = reconstruct_hand(states, player_names or [], button_pos, hero_cards)
    if not hand:
        return ""
    import random
    hid = hand_id or str(random.randint(10000000, 99999999))
    return SnowieExporter(showdown).export(hand, hid)


class SnowieExporter:
    """Exports HandHistory to Snowie/Freezeout format."""

    def __init__(self, showdown: Optional[ShowdownConfig] = None):
        self.showdown = showdown

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
        self._write_actions(f, hand.actions[Street.PREFLOP])

        # Post-flop streets
        if hand.flop_cards:
            f.write(f"FLOP Community Cards:[{' '.join(c.short_name for c in hand.flop_cards)}]\n")
            self._write_actions(f, hand.actions[Street.FLOP])

        if hand.turn_card:
            board = [c.short_name for c in hand.flop_cards] + [hand.turn_card.short_name]
            f.write(f"TURN Community Cards:[{' '.join(board)}]\n")
            self._write_actions(f, hand.actions[Street.TURN])

        if hand.river_card:
            board = [c.short_name for c in hand.flop_cards]
            board += [hand.turn_card.short_name, hand.river_card.short_name]
            f.write(f"RIVER Community Cards:[{' '.join(board)}]\n")
            self._write_actions(f, hand.actions[Street.RIVER])

        # Ending
        if hand.hero_went_all_in:
            self._write_hero_all_in(f, hand, hero_name, opponents)
        elif hand.opponents_folded:
            # All opponents folded, hero wins
            self._write_fold_winner(f, hand)
        elif hand.hero_folded and not hand.reached_showdown:
            # Hero folded and only one opponent left - they win
            self._write_fold_winner(f, hand)
        elif hand.reached_showdown or hand.hero_folded:
            # Multiple players to showdown (hero may or may not be included)
            self._write_showdown(f, hand, hero_name, opponents)

        f.write("GameEnd\n\n")

    def _write_actions(self, f: TextIO, actions: List[HandAction]):
        for a in actions:
            if a.action == PlayerAction.FOLD:
                f.write(f"Move: {a.player_name} folds 0\n")
            elif a.action in (PlayerAction.CHECK, PlayerAction.CALL):
                f.write(f"Move: {a.player_name} call_check {a.amount or 0}\n")
            elif a.action == PlayerAction.UNCALLED_BET:
                f.write(f"uncalled_bet: {a.player_name} {a.amount or 0}\n")
            else:
                f.write(f"Move: {a.player_name} raise_bet {a.amount or 0}\n")

    def _write_hero_all_in(self, f: TextIO, hand: HandHistory, hero_name: str, opponents):
        for p in opponents:
            f.write(f"Move: {p.name} folds 0\n")
        f.write(f"Winner: {hero_name} {hand.pot:.2f}\n")

    def _write_fold_winner(self, f: TextIO, hand: HandHistory):
        """Someone folded - write winner (uncalled_bet already written in actions)."""
        f.write(f"Winner: {hand.winner} {hand.payout:.2f}\n")

    def _write_showdown(self, f: TextIO, hand: HandHistory, hero_name: str, opponents):
        """Write showdown when hand reaches showdown with multiple players."""
        from .game_simulator import make_showdown_config

        # Find opponents who folded (excluded from showdown)
        folded_players = set()
        for street_actions in hand.actions.values():
            for a in street_actions:
                if a.action == PlayerAction.FOLD:
                    folded_players.add(a.player_name)

        # Active opponents (didn't fold)
        active_opponents = [p for p in opponents if p.name not in folded_players]

        community = []
        if hand.flop_cards:
            community = [c.short_name for c in hand.flop_cards]
        if hand.turn_card:
            community.append(hand.turn_card.short_name)
        if hand.river_card:
            community.append(hand.river_card.short_name)

        hero_cards = [c.short_name for c in hand.hero_cards] if hand.hero_cards else []

        # Generate showdown config if not provided (only for active opponents)
        showdown = self.showdown
        if not showdown or not showdown.opponent_cards:
            opponent_names = [p.name for p in active_opponents]
            showdown = make_showdown_config(hero_cards, community, opponent_names)

        # Filter to only active opponents
        active_opponent_cards = {
            name: cards for name, cards in showdown.opponent_cards.items()
            if name not in folded_players
        }

        # Build hands for winner determination
        all_hands = dict(active_opponent_cards)
        if hero_cards:
            all_hands[hero_name] = hero_cards

        # Write showdown for hero
        if hand.hero_cards:
            f.write(f"Showdown: {hero_name} [{' '.join(hero_cards)}]\n")

        # Write showdown for active opponents only
        for name, cards in active_opponent_cards.items():
            f.write(f"Showdown: {name} [{' '.join(cards)}]\n")

        # Determine winner
        if showdown.force_winner and showdown.force_winner not in folded_players:
            winner = showdown.force_winner
        else:
            winner, _ = pick_winner(all_hands, community)
        f.write(f"Winner: {winner} {hand.pot:.2f}\n")
