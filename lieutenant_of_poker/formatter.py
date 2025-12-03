"""
Output formatting for game state analysis.
"""

from typing import Dict, List, Optional
from .game_state import GameState


def format_changes(
    states: List[GameState],
    verbose: bool = False,
    player_names: Optional[Dict[int, str]] = None,
    num_players: Optional[int] = None,
) -> str:
    """Format states as first frame info + list of changes.

    Args:
        states: List of game states to format.
        verbose: If True, show [X] prefix on rejected states.
        player_names: Optional mapping of player index to detected player name.
        num_players: Number of players (hero is always last). If None, inferred from states.
    """
    # Infer number of players from first state if not provided
    if num_players is None and states:
        num_players = len(states[0].players)
    hero_index = num_players - 1 if num_players else None

    def get_name(pos: int) -> str:
        if player_names and pos in player_names and player_names[pos]:
            return player_names[pos]
        if hero_index is not None and pos == hero_index:
            return "Hero"
        return f"Player {pos}"
    if not states:
        return "No frames analyzed."

    lines = []
    first = states[0]

    # First frame: full info
    lines.append(f"=== Frame {first.frame_number} ({first.timestamp_ms:.0f}ms) ===")
    lines.append(f"Street: {first.street.name}")
    lines.append(f"Pot: {first.pot}")
    lines.append(f"Community: {' '.join(str(c) for c in first.community_cards) or '-'}")
    lines.append(f"Hero: {' '.join(str(c) for c in first.hero_cards) or '-'}")
    for pos, player in first.players.items():
        lines.append(f"  {get_name(pos)}: {player.chips}")

    # Track previous accepted state for computing deltas
    prev = first

    # Subsequent frames: only changes
    for state in states[1:]:
        changes = []

        if state.pot != prev.pot:
            changes.append(f"pot: {prev.pot} → {state.pot}")

        if state.street != prev.street:
            changes.append(f"street: {prev.street.name} → {state.street.name}")

        prev_community = [str(c) for c in prev.community_cards]
        curr_community = [str(c) for c in state.community_cards]
        if curr_community != prev_community:
            changes.append(f"community: {' '.join(curr_community)}")

        prev_hero = [str(c) for c in prev.hero_cards]
        curr_hero = [str(c) for c in state.hero_cards]
        if curr_hero != prev_hero:
            changes.append(f"hero: {' '.join(curr_hero)}")

        total_bet = 0
        for pos in state.players:
            prev_chips = prev.players.get(pos)
            curr_chips = state.players.get(pos)
            if prev_chips and curr_chips and prev_chips.chips != curr_chips.chips:
                delta = curr_chips.chips - prev_chips.chips
                if delta < 0:
                    total_bet += -delta
                changes.append(f"{get_name(pos)}: {prev_chips.chips} → {curr_chips.chips}")

        if total_bet > 0 and not state.rejected:
            changes.insert(0, f"bet: {total_bet}")

        if changes:
            prefix = "[X] " if (state.rejected and verbose) else ""
            lines.append(f"{prefix}[{state.frame_number}] {state.timestamp_ms:.0f}ms: {', '.join(changes)}")

            # Only update prev for valid (non-rejected) transitions
            if not state.rejected:
                prev = state
        else:
            if not state.rejected:
                prev = state

    return '\n'.join(lines)
