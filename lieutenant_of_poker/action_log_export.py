"""Export hand history as a simple chronological action log."""

# Street constants
PREFLOP = "preflop"
FLOP = "flop"
TURN = "turn"
RIVER = "river"

# Action constants
FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"
BET = "bet"
ALL_IN = "all_in"
UNCALLED_BET = "uncalled_bet"


def format_action_log(hand: dict) -> str:
    """Format a hand dict as a simple chronological action log."""
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
    if act == FOLD:
        return f"{a['player_name']} folds"
    elif act == CHECK:
        return f"{a['player_name']} checks"
    elif act == CALL:
        return f"{a['player_name']} calls ${a['amount']}"
    elif act == RAISE:
        return f"{a['player_name']} raises to ${a['amount']}"
    elif act == BET:
        return f"{a['player_name']} bets ${a['amount']}"
    elif act == ALL_IN:
        return f"{a['player_name']} goes all-in ${a['amount']}"
    elif act == UNCALLED_BET:
        return f"Uncalled bet (${a['amount']}) returned to {a['player_name']}"
    else:
        return f"{a['player_name']} acts"
