"""Export hand history in human-readable format."""

import io

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


def format_human(hand: dict) -> str:
    """Format a hand dict in human-readable format."""
    output = io.StringIO()
    f = output

    # Header
    f.write(f"Stakes: ${hand['small_blind']}/${hand['big_blind']}\n")
    f.write(f"Table: {hand['table_name']}\n\n")

    # Players
    f.write("Players:\n")
    for i, p in enumerate(hand["players"]):
        role = ""
        if i == hand["button_seat"]:
            role = " (BTN)"
        if i == hand["sb_seat"]:
            role = " (SB)"
        if i == hand["bb_seat"]:
            role = " (BB)"
        hero_mark = " *" if i == len(hand["players"]) - 1 else ""
        f.write(f"  Seat {i + 1}: {p['name']} - ${p['chips']}{role}{hero_mark}\n")

    # Blinds
    sb = hand["players"][hand["sb_seat"]]
    bb = hand["players"][hand["bb_seat"]]
    f.write(f"\n{sb['name']} posts small blind ${hand['small_blind']}\n")
    f.write(f"{bb['name']} posts big blind ${hand['big_blind']}\n")

    # Hero cards
    hero = hand["players"][-1]
    if hand["hero_cards"]:
        cards = " ".join(hand["hero_cards"])
        f.write(f"\n{hero['name']} is dealt: [{cards}]\n")

    # Preflop
    f.write("\n--- PREFLOP ---\n")
    _write_actions(f, hand["actions"][PREFLOP])

    # Flop
    if hand["flop_cards"]:
        cards = " ".join(hand["flop_cards"])
        f.write(f"\n--- FLOP [{cards}] ---\n")
        _write_actions(f, hand["actions"][FLOP])

    # Turn
    if hand["turn_card"]:
        flop = " ".join(hand["flop_cards"])
        f.write(f"\n--- TURN [{flop} {hand['turn_card']}] ---\n")
        _write_actions(f, hand["actions"][TURN])

    # River
    if hand["river_card"]:
        board = " ".join(hand["flop_cards"])
        board += f" {hand['turn_card']} {hand['river_card']}"
        f.write(f"\n--- RIVER [{board}] ---\n")
        _write_actions(f, hand["actions"][RIVER])

    # Ending
    f.write("\n--- RESULT ---\n")
    if hand["reached_showdown"]:
        f.write("Went to showdown\n")
    elif hand["hero_folded"]:
        f.write("Hero folded\n")
    elif hand["opponents_folded"]:
        f.write("Opponents folded\n")
    f.write(f"Final pot: ${hand['pot']}\n")

    return output.getvalue()


def _write_actions(f, actions: list[dict]):
    if not actions:
        f.write("  (no actions)\n")
        return

    for a in actions:
        act = a["action"]
        if act == FOLD:
            f.write(f"  {a['player_name']} folds\n")
        elif act == CHECK:
            f.write(f"  {a['player_name']} checks\n")
        elif act == CALL:
            f.write(f"  {a['player_name']} calls ${a['amount']}\n")
        elif act == RAISE:
            f.write(f"  {a['player_name']} raises to ${a['amount']}\n")
        elif act == BET:
            f.write(f"  {a['player_name']} bets ${a['amount']}\n")
        elif act == ALL_IN:
            f.write(f"  {a['player_name']} is all-in for ${a['amount']}\n")
        else:
            f.write(f"  {a['player_name']} unknown action\n")
