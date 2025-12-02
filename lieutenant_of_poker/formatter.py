"""
Output formatting for game state analysis.
"""

from typing import List, Optional
from .game_state import GameState


def format_changes(states: List[GameState], validate: bool = False) -> str:
    """Format states as first frame info + list of changes.

    Args:
        states: List of game states to format.
        validate: If True, mark changes that would be rejected by the validator.
    """
    if not states:
        return "No frames analyzed."

    validator = None
    if validate:
        from .rules_validator import RulesValidator
        validator = RulesValidator(allow_new_hand=False, check_chip_increases=True)

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
            # Check if this transition would be rejected
            rejected = False
            if validator:
                result = validator.validate_transition(prev, state)
                rejected = not result.is_valid

            prefix = "[X] " if rejected else ""
            lines.append(f"{prefix}[{state.frame_number}] {state.timestamp_ms:.0f}ms: {', '.join(changes)}")

            # Only update prev for valid transitions
            if not rejected:
                prev = state
        else:
            prev = state

    return '\n'.join(lines)
