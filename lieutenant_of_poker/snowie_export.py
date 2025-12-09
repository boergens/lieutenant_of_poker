"""Export hand history to PokerSnowie format."""

import io
import random

from .game_simulator import ShowdownConfig, pick_winner, make_showdown_config

# Street constants
PREFLOP = "preflop"
FLOP = "flop"
TURN = "turn"
RIVER = "river"

# Action constants
FOLD = "fold"
CHECK = "check"
CALL = "call"
UNCALLED_BET = "uncalled_bet"


def format_snowie(hand: dict, hand_id: str | None = None, showdown: ShowdownConfig | None = None) -> str:
    """Format a hand dict in Snowie format.

    Args:
        hand: Hand history dict from reconstruct_hand
        hand_id: Optional hand ID (random 8-digit number if not provided)
        showdown: Showdown configuration with opponent cards for deterministic output.
    """
    hid = hand_id or str(random.randint(10000000, 99999999))

    output = io.StringIO()
    f = output

    # Hero is always the last player in the list
    hero = hand["players"][-1]
    hero_name = hero["name"]
    opponents = hand["players"][:-1]
    sb = hand["players"][hand["sb_seat"]]
    bb = hand["players"][hand["bb_seat"]]

    # Header
    f.write("GameStart\n")
    f.write("PokerClient: ExportFormat\n")
    f.write(f"Date: {hand['timestamp'].strftime('%d/%m/%Y')}\n")
    f.write("TimeZone: GMT\n")
    f.write(f"Time: {hand['timestamp'].strftime('%H:%M:%S')}\n")
    f.write(f"GameId:{hid}\n")
    f.write("GameType:NoLimit\n")
    f.write("GameCurrency: $\n")
    f.write(f"SmallBlindStake: {hand['small_blind']}\n")
    f.write(f"BigBlindStake: {hand['big_blind']}\n")
    f.write("AnteStake: 0\n")
    f.write(f"TableName: {hand['table_name']}\n")
    f.write(f"Max number of players: {len(hand['players'])}\n")
    f.write(f"MyPlayerName: {hero_name}\n")
    # Heads-up: dealer position is inverted in Snowie format
    if len(hand["players"]) == 2:
        dealer_pos = 1 - hand["button_seat"]
    else:
        dealer_pos = hand["button_seat"]
    f.write(f"DealerPosition: {dealer_pos}\n")

    for i, p in enumerate(hand["players"]):
        f.write(f"Seat {i} {p['name']} {p['chips']}\n")

    f.write(f"SmallBlind: {sb['name']} {hand['small_blind']}\n")
    f.write(f"BigBlind: {bb['name']} {hand['big_blind']}\n")

    if hand["hero_cards"]:
        f.write(f"Dealt Cards: [{''.join(hand['hero_cards'])}]\n")

    # Preflop
    _write_actions(f, hand["actions"][PREFLOP])

    # Post-flop streets
    if hand["flop_cards"]:
        f.write(f"FLOP Community Cards:[{' '.join(hand['flop_cards'])}]\n")
        _write_actions(f, hand["actions"][FLOP])

    if hand["turn_card"]:
        board = hand["flop_cards"] + [hand["turn_card"]]
        f.write(f"TURN Community Cards:[{' '.join(board)}]\n")
        _write_actions(f, hand["actions"][TURN])

    if hand["river_card"]:
        board = hand["flop_cards"] + [hand["turn_card"], hand["river_card"]]
        f.write(f"RIVER Community Cards:[{' '.join(board)}]\n")
        _write_actions(f, hand["actions"][RIVER])

    # Ending
    if hand["hero_went_all_in"]:
        _write_hero_all_in(f, hand, hero_name, opponents)
    elif hand["opponents_folded"]:
        # All opponents folded, hero wins
        _write_fold_winner(f, hand)
    elif hand["hero_folded"] and not hand["reached_showdown"]:
        # Hero folded and only one opponent left - they win
        _write_fold_winner(f, hand)
    elif hand["reached_showdown"] or hand["hero_folded"]:
        # Multiple players to showdown (hero may or may not be included)
        _write_showdown(f, hand, hero_name, opponents, showdown)

    f.write("GameEnd\n\n")
    return output.getvalue()


def _write_actions(f, actions: list[dict]):
    for a in actions:
        act = a["action"]
        if act == FOLD:
            f.write(f"Move: {a['player_name']} folds 0\n")
        elif act in (CHECK, CALL):
            f.write(f"Move: {a['player_name']} call_check {a['amount'] or 0}\n")
        elif act == UNCALLED_BET:
            f.write(f"uncalled_bet: {a['player_name']} {a['amount'] or 0}\n")
        else:
            f.write(f"Move: {a['player_name']} raise_bet {a['amount'] or 0}\n")


def _write_hero_all_in(f, hand: dict, hero_name: str, opponents: list[dict]):
    for p in opponents:
        f.write(f"Move: {p['name']} folds 0\n")
    f.write(f"Winner: {hero_name} {hand['pot']:.2f}\n")


def _write_fold_winner(f, hand: dict):
    """Someone folded - write winner (uncalled_bet already written in actions)."""
    f.write(f"Winner: {hand['winner']} {hand['payout']:.2f}\n")


def _write_showdown(f, hand: dict, hero_name: str, opponents: list[dict], showdown: ShowdownConfig | None):
    """Write showdown when hand reaches showdown with multiple players."""
    # Find opponents who folded (excluded from showdown)
    folded_players = set()
    for street_actions in hand["actions"].values():
        for a in street_actions:
            if a["action"] == FOLD:
                folded_players.add(a["player_name"])

    # Active opponents (didn't fold)
    active_opponents = [p for p in opponents if p["name"] not in folded_players]

    community = []
    if hand["flop_cards"]:
        community = list(hand["flop_cards"])
    if hand["turn_card"]:
        community.append(hand["turn_card"])
    if hand["river_card"]:
        community.append(hand["river_card"])

    hero_cards = list(hand["hero_cards"]) if hand["hero_cards"] else []

    # Generate showdown config if not provided (only for active opponents)
    if not showdown or not showdown.opponent_cards:
        opponent_names = [p["name"] for p in active_opponents]
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
    if hand["hero_cards"]:
        f.write(f"Showdown: {hero_name} [{' '.join(hero_cards)}]\n")

    # Write showdown for active opponents only
    for name, cards in active_opponent_cards.items():
        f.write(f"Showdown: {name} [{' '.join(cards)}]\n")

    # Determine winner
    if showdown.force_winner and showdown.force_winner not in folded_players:
        winner = showdown.force_winner
    else:
        winner, _ = pick_winner(all_hands, community)
    f.write(f"Winner: {winner} {hand['pot']:.2f}\n")
