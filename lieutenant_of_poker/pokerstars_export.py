"""Export hand history to PokerStars format."""

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


def _calculate_contributions(hand: dict) -> dict:
    """Calculate how much each player put in the pot."""
    contributions = {}
    for p in hand["players"]:
        contributions[p["name"]] = 0

    # Add blinds
    sb_name = hand["players"][hand["sb_seat"]]["name"]
    bb_name = hand["players"][hand["bb_seat"]]["name"]
    contributions[sb_name] = hand["small_blind"]
    contributions[bb_name] = hand["big_blind"]

    # Add actions from all streets
    for street in [PREFLOP, FLOP, TURN, RIVER]:
        for action in hand["actions"].get(street, []):
            name = action["player_name"]
            act = action["action"]
            amount = action.get("amount", 0) or 0
            if act in [CALL, BET, RAISE, ALL_IN]:
                contributions[name] = amount  # Raises/bets are cumulative totals

    return contributions


def _find_winner(hand: dict) -> tuple:
    """Find winner and calculate collected amount. Returns (winner_name, winner_seat, collected)."""
    # If winner is specified in hand dict, use it
    if "winner_seat" in hand:
        winner_seat = hand["winner_seat"]
        winner_name = hand["players"][winner_seat]["name"]
        collected = hand.get("collected", hand["pot"])
        return winner_name, winner_seat, collected

    # Otherwise, find the last player who didn't fold
    folded = set()
    for street in [PREFLOP, FLOP, TURN, RIVER]:
        for action in hand["actions"].get(street, []):
            if action["action"] == FOLD:
                folded.add(action["player_name"])

    for i, p in enumerate(hand["players"]):
        if p["name"] not in folded:
            return p["name"], i, hand["pot"]

    # Fallback to BB
    bb_seat = hand["bb_seat"]
    return hand["players"][bb_seat]["name"], bb_seat, hand["pot"]


def _get_last_aggressor(hand: dict) -> tuple:
    """Find the last player who bet/raised and the uncalled amount. Returns (name, uncalled_amount)."""
    contributions = _calculate_contributions(hand)
    max_contribution = max(contributions.values())

    # Find who has max and if it's uncalled
    callers_at_max = [name for name, amt in contributions.items() if amt == max_contribution]

    if len(callers_at_max) == 1:
        # Uncalled bet - find second highest
        second_max = max(amt for name, amt in contributions.items() if name != callers_at_max[0])
        uncalled = max_contribution - second_max
        if uncalled > 0:
            return callers_at_max[0], uncalled

    return None, 0


def _get_seat_summary(hand: dict, seat: int, winner_seat: int, collected: int) -> str:
    """Generate seat summary line for a player."""
    player = hand["players"][seat]
    name = player["name"]

    # Determine position
    position = ""
    if seat == hand["button_seat"]:
        position = " (button)"
    elif seat == hand["sb_seat"]:
        position = " (small blind)"
    elif seat == hand["bb_seat"]:
        position = " (big blind)"

    # Check if player won
    if seat == winner_seat:
        return f"Seat {seat + 1}: {name}{position} collected (${collected})"

    # Check when/how player folded
    last_street_with_action = None
    for street in [PREFLOP, FLOP, TURN, RIVER]:
        for action in hand["actions"].get(street, []):
            if action["player_name"] == name:
                if action["action"] == FOLD:
                    street_name = "Flop" if street == PREFLOP else street.capitalize()
                    if street == PREFLOP:
                        # Check if they put money in
                        if seat in [hand["sb_seat"], hand["bb_seat"]]:
                            return f"Seat {seat + 1}: {name}{position} folded before Flop"
                        return f"Seat {seat + 1}: {name}{position} folded before Flop (didn't bet)"
                    return f"Seat {seat + 1}: {name}{position} folded on the {street_name}"
                last_street_with_action = street

    return f"Seat {seat + 1}: {name}{position} folded before Flop (didn't bet)"


def format_pokerstars(hand: dict, hand_id: str = "00000000", timestamp: str = None) -> str:
    """Format a hand dict in PokerStars format."""
    output = io.StringIO()
    f = output

    sb = hand["players"][hand["sb_seat"]]
    bb = hand["players"][hand["bb_seat"]]

    # Use provided timestamp, or from hand dict, or default
    ts = timestamp or hand.get("timestamp", "2024/01/01 12:00:00")

    # Header
    f.write(f"PokerStars Hand #{hand_id}: Hold'em No Limit ")
    f.write(f"(${hand['small_blind']}/${hand['big_blind']} USD) - {ts} ET\n")
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

    # Calculate winner and uncalled bet
    winner_name, winner_seat, _ = _find_winner(hand)
    uncalled_player, uncalled_amount = _get_last_aggressor(hand)

    # The actual pot is total contributions minus uncalled amount
    contributions = _calculate_contributions(hand)
    total_contributed = sum(contributions.values())
    actual_pot = total_contributed - uncalled_amount

    # Uncalled bet line (before summary)
    if uncalled_amount > 0:
        f.write(f"Uncalled bet (${uncalled_amount}) returned to {uncalled_player}\n")

    # Collection line (before summary)
    f.write(f"{winner_name} collected ${actual_pot} from pot\n")

    # Summary
    f.write("*** SUMMARY ***\n")
    f.write(f"Total pot ${actual_pot} | Rake $0\n")

    board_cards = list(hand["flop_cards"]) if hand["flop_cards"] else []
    if hand["turn_card"]:
        board_cards.append(hand["turn_card"])
    if hand["river_card"]:
        board_cards.append(hand["river_card"])
    if board_cards:
        board = " ".join(board_cards)
        f.write(f"Board [{board}]\n")

    # Seat summaries
    for i in range(len(hand["players"])):
        f.write(f"{_get_seat_summary(hand, i, winner_seat, actual_pot)}\n")

    return output.getvalue()


def _format_action(action: dict) -> str:
    name = action["player_name"]
    act = action["action"]
    amount = action["amount"]
    if act == FOLD:
        return f"{name}: folds"
    elif act == CHECK:
        return f"{name}: checks"
    elif act == CALL:
        return f"{name}: calls ${amount}" if amount else f"{name}: calls"
    elif act == RAISE:
        return f"{name}: raises to ${amount}" if amount else f"{name}: raises"
    elif act == BET:
        return f"{name}: bets ${amount}" if amount else f"{name}: bets"
    elif act == ALL_IN:
        return f"{name}: raises to ${amount} and is all-in" if amount else f"{name}: is all-in"
    return f"{name}: unknown action"
