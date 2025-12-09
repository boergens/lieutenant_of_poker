"""Export hand history as a simple chronological action log."""

from .export import PREFLOP, FLOP, TURN, RIVER, reconstruct_hand
from .action_detector import PlayerAction


def export_action_log(
    states: list[dict],
    button_pos: int | None = None,
    player_names: list[str] | None = None,
) -> str:
    """Export game states as a simple chronological action log."""
    hero_cards = []
    for state in states:
        if state.get("hero_cards"):
            hero_cards = state["hero_cards"]
            break
    hand = reconstruct_hand(states, player_names or [], button_pos, hero_cards)
    if not hand:
        return "No hand data."
    return _format_action_log(hand)


def _format_action_log(hand: dict) -> str:
    lines = []

    # Blinds
    sb = hand["players"][hand["sb_seat"]]
    bb = hand["players"][hand["bb_seat"]]
    lines.append(f"{sb['name']} posts small blind ${hand['small_blind']}")
    lines.append(f"{bb['name']} posts big blind ${hand['big_blind']}")

    # Hole cards
    hero = hand["players"][-1]
    if hand["hero_cards"]:
        cards = " ".join(hand["hero_cards"])
        lines.append(f"Dealer deals hole cards to {hero['name']}: [{cards}]")

    # Preflop actions
    for a in hand["actions"][PREFLOP]:
        lines.append(_format_action(a))

    # Flop
    if hand["flop_cards"]:
        cards = " ".join(hand["flop_cards"])
        lines.append(f"Dealer reveals flop: {cards}")
        for a in hand["actions"][FLOP]:
            lines.append(_format_action(a))

    # Turn
    if hand["turn_card"]:
        lines.append(f"Dealer reveals turn: {hand['turn_card']}")
        for a in hand["actions"][TURN]:
            lines.append(_format_action(a))

    # River
    if hand["river_card"]:
        lines.append(f"Dealer reveals river: {hand['river_card']}")
        for a in hand["actions"][RIVER]:
            lines.append(_format_action(a))

    return "\n".join(lines)


def _format_action(a: dict) -> str:
    act = a["action"]
    if act == PlayerAction.FOLD:
        return f"{a['player_name']} folds"
    elif act == PlayerAction.CHECK:
        return f"{a['player_name']} checks"
    elif act == PlayerAction.CALL:
        return f"{a['player_name']} calls ${a['amount']}"
    elif act == PlayerAction.RAISE:
        return f"{a['player_name']} raises to ${a['amount']}"
    elif act == PlayerAction.BET:
        return f"{a['player_name']} bets ${a['amount']}"
    elif act == PlayerAction.ALL_IN:
        return f"{a['player_name']} goes all-in ${a['amount']}"
    elif act == PlayerAction.UNCALLED_BET:
        return f"Uncalled bet (${a['amount']}) returned to {a['player_name']}"
    else:
        return f"{a['player_name']} acts"
