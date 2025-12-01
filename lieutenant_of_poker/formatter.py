"""
Output formatting for game state analysis.
"""

from typing import List
from .game_state import GameState


def format_changes(states: List[GameState]) -> str:
    """Format states as first frame info + list of changes."""
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
        lines.append(f"  {pos.name}: {player.chips}")

    # Track previous state
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

        for pos in state.players:
            prev_chips = prev.players.get(pos)
            curr_chips = state.players.get(pos)
            if prev_chips and curr_chips and prev_chips.chips != curr_chips.chips:
                changes.append(f"{pos.name}: {prev_chips.chips} → {curr_chips.chips}")

        if changes:
            lines.append(f"[{state.frame_number}] {state.timestamp_ms:.0f}ms: {', '.join(changes)}")

        prev = state

    return '\n'.join(lines)
