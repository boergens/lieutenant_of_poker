"""Export hand history to PokerStars format."""

import io

from .export import PREFLOP, FLOP, TURN, RIVER, reconstruct_hand, calculate_pot
from .action_detector import PlayerAction


def export_pokerstars(
    states: list[dict],
    button_pos: int = 0,
    player_names: list[str] | None = None,
    hand_id: str | None = None,
) -> str:
    """Export game states to PokerStars format."""
    hero_cards = []
    for state in states:
        if state.get("hero_cards"):
            hero_cards = state["hero_cards"]
            break
    hand = reconstruct_hand(states, player_names or [], button_pos, hero_cards)
    if not hand:
        return ""
    return _format_pokerstars(hand, hand_id or "00000000")


def _format_pokerstars(hand: dict, hand_id: str) -> str:
    output = io.StringIO()
    f = output

    sb = hand["players"][hand["sb_seat"]]
    bb = hand["players"][hand["bb_seat"]]

    # Header
    f.write(f"PokerStars Hand #{hand_id}: Hold'em No Limit ")
    f.write(f"(${hand['small_blind']}/${hand['big_blind']})\n")
    f.write(f"Table '{hand['table_name']}' 6-max Seat #{hand['button_seat'] + 1} is the button\n")

    # Players
    for i, p in enumerate(hand["players"]):
        f.write(f"Seat {i + 1}: {p['name']} (${p['chips']} in chips)\n")

    # Blinds
    f.write(f"{sb['name']}: posts small blind ${hand['small_blind']}\n")
    f.write(f"{bb['name']}: posts big blind ${hand['big_blind']}\n")

    # Hole cards
    hero = hand["players"][-1]
    f.write("*** HOLE CARDS ***\n")
    if hand["hero_cards"]:
        cards = " ".join(hand["hero_cards"])
        f.write(f"Dealt to {hero['name']} [{cards}]\n")

    # Preflop
    for a in hand["actions"][PREFLOP]:
        f.write(f"{_format_action(a)}\n")

    # Flop
    if hand["flop_cards"]:
        cards = " ".join(hand["flop_cards"])
        f.write(f"*** FLOP *** [{cards}]\n")
        for a in hand["actions"][FLOP]:
            f.write(f"{_format_action(a)}\n")

    # Turn
    if hand["turn_card"]:
        flop = " ".join(hand["flop_cards"])
        f.write(f"*** TURN *** [{flop}] [{hand['turn_card']}]\n")
        for a in hand["actions"][TURN]:
            f.write(f"{_format_action(a)}\n")

    # River
    if hand["river_card"]:
        board = " ".join(hand["flop_cards"])
        board += f" {hand['turn_card']}"
        f.write(f"*** RIVER *** [{board}] [{hand['river_card']}]\n")
        for a in hand["actions"][RIVER]:
            f.write(f"{_format_action(a)}\n")

    # Summary
    f.write("*** SUMMARY ***\n")
    f.write(f"Total pot ${calculate_pot(hand)}\n")

    board_cards = list(hand["flop_cards"]) if hand["flop_cards"] else []
    if hand["turn_card"]:
        board_cards.append(hand["turn_card"])
    if hand["river_card"]:
        board_cards.append(hand["river_card"])
    if board_cards:
        board = " ".join(board_cards)
        f.write(f"Board [{board}]\n")

    return output.getvalue()


def _format_action(action: dict) -> str:
    name = action["player_name"]
    act = action["action"]
    amount = action["amount"]
    if act == PlayerAction.FOLD:
        return f"{name}: folds"
    elif act == PlayerAction.CHECK:
        return f"{name}: checks"
    elif act == PlayerAction.CALL:
        return f"{name}: calls ${amount}" if amount else f"{name}: calls"
    elif act == PlayerAction.RAISE:
        return f"{name}: raises to ${amount}" if amount else f"{name}: raises"
    elif act == PlayerAction.BET:
        return f"{name}: bets ${amount}" if amount else f"{name}: bets"
    elif act == PlayerAction.ALL_IN:
        return f"{name}: raises to ${amount} and is all-in" if amount else f"{name}: is all-in"
    return f"{name}: unknown action"
