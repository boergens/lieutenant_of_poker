"""
Hand history reconstruction from game state observations.

Shared data structures and reconstruction logic used by all exporters.
"""

from datetime import datetime
from typing import Any


# Street constants
PREFLOP = "preflop"
FLOP = "flop"
TURN = "turn"
RIVER = "river"
STREETS = [PREFLOP, FLOP, TURN, RIVER]

# Action constants
FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"
BET = "bet"
ALL_IN = "all_in"
UNCALLED_BET = "uncalled_bet"

# Common blind structures: (small_blind, big_blind)
BLIND_STRUCTURES = [
    (1, 2),
    (2, 5),
    (5, 10),
    (10, 20),
    (15, 30),
    (20, 40),
    (25, 50),
    (50, 100),
    (75, 150),
    (100, 200),
    (150, 300),
    (200, 400),
    (250, 500),
    (300, 600),
    (400, 800),
    (500, 1000),
]


def derive_blinds(pot: int) -> tuple[int, int]:
    """Derive small and big blind from initial pot size.

    Finds the blind structure where SB + BB equals the pot,
    or falls back to assuming pot = SB + BB with BB = 2*SB.
    """
    if not pot:
        return 10, 20

    # Check known blind structures
    for sb, bb in BLIND_STRUCTURES:
        if sb + bb == pot:
            return sb, bb

    # Fallback: assume BB = 2*SB, so pot = 3*SB
    sb = pot // 3
    bb = sb * 2
    return sb, bb


def calculate_pot(hand: dict) -> int:
    """Calculate pot from actions (blinds + all bets/calls/raises minus uncalled)."""
    total = hand["small_blind"] + hand["big_blind"]
    for street_actions in hand["actions"].values():
        for action in street_actions:
            if action["action"] in (BET, RAISE, CALL, ALL_IN):
                total += action["amount"] or 0
            elif action["action"] == UNCALLED_BET:
                total -= action["amount"] or 0
    return total


def make_hand_history() -> dict:
    """Create an empty hand history dict."""
    return {
        "table_name": "Governor of Poker",
        "timestamp": datetime.now(),
        "small_blind": 10,
        "big_blind": 20,
        "players": [],
        "button_seat": 0,
        "sb_seat": 0,
        "bb_seat": 0,
        "hero_cards": [],
        "flop_cards": None,
        "turn_card": None,
        "river_card": None,
        "actions": {
            PREFLOP: [],
            FLOP: [],
            TURN: [],
            RIVER: [],
        },
        "hero_went_all_in": False,
        "hero_folded": False,
        "opponents_folded": False,
        "reached_showdown": False,
        "winner": None,
        "payout": 0,
    }


def reconstruct_hand(states: list[dict], table) -> dict | None:
    """Reconstruct a hand history from a sequence of game state observations.

    Args:
        states: List of game state dicts representing the hand progression
        table: TableInfo with player names, button position, hero cards

    Returns:
        Hand history dict if reconstruction succeeds, None otherwise

    Note:
        The hero is always the last player in the list.
    """
    if not states:
        return None

    players = list(table.names)
    button_pos = table.button_index or 0
    hero_cards = list(table.hero_cards)

    initial, final = states[0], states[-1]

    # Derive blinds from initial pot
    small_blind, big_blind = derive_blinds(initial["pot"])

    if not players:
        return None

    # Calculate blind positions first (heads-up: button = SB)
    if len(players) == 2:
        sb_idx, bb_idx = button_pos, (button_pos + 1) % 2
    else:
        sb_idx = (button_pos + 1) % len(players)
        bb_idx = (button_pos + 2) % len(players)

    # Build player info list
    # Add blinds back since initial state shows chips AFTER posting
    player_infos = []
    for i, name in enumerate(players):
        chips = initial["players"][i]["chips"] if i < len(initial["players"]) else 0
        if i == sb_idx:
            chips += small_blind
        elif i == bb_idx:
            chips += big_blind
        player_infos.append({"name": name, "chips": chips, "position": i})

    hand = make_hand_history()
    hand["small_blind"] = small_blind
    hand["big_blind"] = big_blind
    hand["players"] = player_infos
    hand["button_seat"] = button_pos
    hand["sb_seat"] = sb_idx
    hand["bb_seat"] = bb_idx
    hand["hero_cards"] = hero_cards

    # Detect chip movements from frame-to-frame comparison, organized by street
    # This is a dumb pass - just record who put in how much, when
    chip_movements = {s: [] for s in STREETS}

    current_street = PREFLOP
    prev_state = states[0]

    for state in states[1:]:
        # Determine street from community cards
        num_cards = len(state.get("community_cards", []))
        if num_cards >= 5:
            new_street = RIVER
        elif num_cards >= 4:
            new_street = TURN
        elif num_cards >= 3:
            new_street = FLOP
        else:
            new_street = PREFLOP

        if new_street != current_street:
            if new_street == FLOP and num_cards >= 3:
                hand["flop_cards"] = state["community_cards"][:3]
            elif new_street == TURN and num_cards >= 4:
                hand["turn_card"] = state["community_cards"][3]
            elif new_street == RIVER and num_cards >= 5:
                hand["river_card"] = state["community_cards"][4]
            current_street = new_street

        # Detect chip movements by comparing player stacks directly
        # (Don't rely on pot change since rake can reduce the pot)
        for i, curr_p in enumerate(state["players"]):
            if i < len(prev_state["players"]) and i < len(players):
                prev_p = prev_state["players"][i]
                chip_change = (prev_p["chips"] or 0) - (curr_p["chips"] or 0)
                if chip_change > 0:
                    name = players[i]
                    is_all_in = curr_p["chips"] == 0
                    movement = {
                        "player_name": name,
                        "amount": chip_change,
                        "is_all_in": is_all_in,
                    }
                    chip_movements[current_street].append(movement)

        prev_state = state

    # Add blind posts as chip movements at start of preflop
    sb_name = players[sb_idx]
    bb_name = players[bb_idx]
    chip_movements[PREFLOP].insert(0, {"player_name": bb_name, "amount": big_blind, "is_all_in": False})
    chip_movements[PREFLOP].insert(0, {"player_name": sb_name, "amount": small_blind, "is_all_in": False})

    # players_in_hand: who hasn't folded (may include all-in players)
    players_in_hand = list(players)

    def street_reached(street: str) -> bool:
        if street == PREFLOP:
            return True
        elif street == FLOP:
            return hand["flop_cards"] is not None
        elif street == TURN:
            return hand["turn_card"] is not None
        elif street == RIVER:
            return hand["river_card"] is not None
        return False

    # Cumulative set of players who are all-in (added as we process actions)
    players_all_in: set = set()

    def get_next_active_player(start_pos: int, in_hand: list[str], all_in: set) -> tuple[int, str] | None:
        """Find next player who can act, starting from start_pos."""
        for i in range(len(players)):
            pos = (start_pos + i) % len(players)
            name = players[pos]
            if name in in_hand and name not in all_in:
                return pos, name
        return None

    for street in STREETS:
        if not street_reached(street) or len(players_in_hand) <= 1:
            break

        # If all but one player are all-in, no betting action needed on this street
        # The hand proceeds automatically to showdown
        active_players = [p for p in players_in_hand if p not in players_all_in]
        if len(active_players) <= 1:
            continue

        actions_list = hand["actions"][street]
        street_movements = chip_movements[street]

        # Contributions this street (starts empty, built from movements)
        contributions: dict[str, int] = {}
        current_bet = 0

        # First player to act (SB for all streets - preflop has blind posts first)
        first_to_act_idx = sb_idx

        # Current position in the rotation (index into players list)
        current_pos = first_to_act_idx

        # Process each chip movement and classify it based on betting context
        for movement in street_movements:
            if len(players_in_hand) <= 1:
                break

            actor = movement["player_name"]

            # Walk from current position to actor, inferring checks/folds
            while True:
                result = get_next_active_player(current_pos, players_in_hand, players_all_in)
                if result is None:
                    break
                pos, name = result

                if name == actor:
                    # Found the actor - classify and record their action
                    # Now we have full betting context to determine action type
                    new_total = contributions.get(name, 0) + movement["amount"]

                    if new_total > current_bet:
                        action_type = BET if current_bet == 0 else RAISE
                    else:
                        action_type = CALL

                    action = {"player_name": name, "action": action_type, "amount": movement["amount"]}
                    actions_list.append(action)

                    contributions[name] = new_total
                    if new_total > current_bet:
                        current_bet = new_total

                    # Mark as all-in if this movement put them all-in
                    if movement["is_all_in"]:
                        players_all_in.add(name)
                    current_pos = (pos + 1) % len(players)
                    break
                else:
                    # This player acted before the actor - infer check or fold
                    to_call = current_bet - contributions.get(name, 0)
                    if to_call > 0:
                        actions_list.append({"player_name": name, "action": FOLD, "amount": 0})
                        players_in_hand = [p for p in players_in_hand if p != name]
                    else:
                        actions_list.append({"player_name": name, "action": CHECK, "amount": 0})
                    current_pos = (pos + 1) % len(players)

        # After all movements, close out remaining players
        # - Players who owe chips must fold
        # - Players who don't owe (and haven't acted) must check
        if len(players_in_hand) > 1:
            for i in range(len(players)):
                pos = (current_pos + i) % len(players)
                name = players[pos]
                if name not in players_in_hand or name in players_all_in:
                    continue
                to_call = current_bet - contributions.get(name, 0)
                if to_call > 0:
                    actions_list.append({"player_name": name, "action": FOLD, "amount": 0})
                    players_in_hand = [p for p in players_in_hand if p != name]
                elif name not in contributions:
                    # Player hasn't acted yet and doesn't owe anything - they check
                    actions_list.append({"player_name": name, "action": CHECK, "amount": 0})

    # BB option: if action limped to BB preflop, BB gets option to check/raise
    # Count BB's voluntary actions (exclude the forced blind post which is first 2 actions)
    bb_voluntary_actions = [
        a for a in hand["actions"][PREFLOP][2:]  # Skip SB and BB blind posts
        if a["player_name"] == bb_name
    ]
    bb_folded_preflop = any(a["action"] == FOLD for a in bb_voluntary_actions)
    if not bb_folded_preflop and not bb_voluntary_actions and hand["flop_cards"]:
        hand["actions"][PREFLOP].append({"player_name": bb_name, "action": CHECK, "amount": 0})

    # Calculate uncalled bet BEFORE removing synthetic blinds
    # Skip the blind posts (first 2 preflop actions) - blinds are mandatory,
    # not voluntary bets, so they don't count as "uncalled" if everyone folds.
    # But include their contribution amounts for proper totaling.
    all_actions = (
        hand["actions"][PREFLOP][2:] +  # Skip blind posts
        hand["actions"][FLOP] +
        hand["actions"][TURN] +
        hand["actions"][RIVER]
    )

    # Pre-seed contributions with blind amounts (they're not "bets" for uncalled purposes
    # but they do count toward total contributions)
    contributions = {
        sb_name: small_blind,
        bb_name: big_blind,
    }
    last_bettor: str | None = None
    last_bet_total = 0

    for a in all_actions:
        player = a["player_name"]
        amount = a["amount"] or 0

        if a["action"] in (BET, RAISE):
            contributions[player] = contributions.get(player, 0) + amount
            last_bettor = player
            last_bet_total = contributions[player]
        elif a["action"] == CALL:
            contributions[player] = contributions.get(player, 0) + amount

    # Find max contribution from players other than last_bettor
    uncalled_amount = 0
    if last_bettor:
        max_other_contribution = 0
        for player, total in contributions.items():
            if player != last_bettor:
                max_other_contribution = max(max_other_contribution, total)

        if last_bet_total > max_other_contribution:
            uncalled_amount = last_bet_total - max_other_contribution
            # Add to the last street that has actions
            for street in reversed(STREETS):
                if hand["actions"][street]:
                    hand["actions"][street].append(
                        {"player_name": last_bettor, "action": UNCALLED_BET, "amount": uncalled_amount}
                    )
                    break

    # Remove the synthetic blind posts from preflop actions
    del hand["actions"][PREFLOP][:2]

    # Determine outcome
    # Hero is always the last player in the list
    hero_player_name = players[-1]
    hand["reached_showdown"] = len(players_in_hand) > 1
    hand["opponents_folded"] = len(players_in_hand) == 1 and hero_player_name in players_in_hand
    hand["hero_folded"] = hero_player_name not in players_in_hand

    # Calculate final pot
    hand["pot"] = calculate_pot(hand)

    # Set winner if everyone folded
    if len(players_in_hand) == 1:
        hand["winner"] = players_in_hand[0]
        hand["payout"] = hand["pot"] - uncalled_amount

    return hand


def export_video(
    video_path: str,
    fmt: str,
    button: int | None = None,
    max_rake_pct: float = 0.10,
) -> str | None:
    """
    Analyze a video and export to the specified format.

    Args:
        video_path: Path to the video file.
        fmt: Export format (pokerstars, snowie, human, actions).
        button: Button position override (None = auto-detect).
        max_rake_pct: Maximum rake as percentage of pot (default 10%, 0 to disable).

    Returns:
        Formatted output string, or None if no states found.
    """
    from .analysis import analyze_video
    from .first_frame import TableInfo
    from .snowie_export import format_snowie
    from .pokerstars_export import format_pokerstars
    from .human_export import format_human
    from .action_log_export import format_action_log

    table = TableInfo.from_video(video_path)

    states = analyze_video(video_path, max_rake_pct=max_rake_pct)
    if not states:
        return None

    hand = reconstruct_hand(states, table)
    if not hand:
        return None

    if fmt == "snowie":
        return format_snowie(hand)
    elif fmt == "human":
        return format_human(hand)
    elif fmt == "actions":
        return format_action_log(hand)
    else:
        return format_pokerstars(hand)
