"""
Video analysis for Governor of Poker.

Extracts game states from video frames using TableInfo for player positions.
"""

import sys
from collections import Counter
from typing import Optional

from .first_frame import TableInfo
from .frame_extractor import VideoFrameExtractor
from .card_matcher import match_community_cards
from .chip_ocr import extract_pot, extract_player_money, clear_caches

# Number of consecutive matching frames needed to accept a state change
CONSENSUS_FRAMES = 3


def _state_key(state: dict) -> tuple:
    """Create a hashable key from state for comparison."""
    return (
        tuple(state["community_cards"]),
        state["pot"],
        tuple(p["chips"] for p in state["players"]),
    )


def _states_equivalent(s1: dict, s2: dict) -> bool:
    """Check if two states have the same values."""
    return _state_key(s1) == _state_key(s2)


def _is_complete(state: dict) -> bool:
    """Check if state has all required values (no None in critical fields)."""
    if state["pot"] is None:
        return False
    # All players must have chips
    return all(p["chips"] is not None for p in state["players"])


def _total_chips(state: dict) -> int | None:
    """Calculate total chips in circulation (pot + all player chips)."""
    total = 0
    if state["pot"] is None:
        return None
    total += state["pot"]
    for p in state["players"]:
        if p["chips"] is None:
            return None
        total += p["chips"]
    return total


def _validate_transition(prev: dict, curr: dict) -> tuple[bool, list[str]]:
    """
    Validate a state transition.

    Returns (is_valid, list of violation messages).
    """
    violations = []

    # Community cards can't decrease
    if len(curr["community_cards"]) < len(prev["community_cards"]):
        violations.append(f"community cards decreased: {len(prev['community_cards'])} → {len(curr['community_cards'])}")

    # Community cards must be valid count (0, 3, 4, or 5)
    if len(curr["community_cards"]) not in (0, 3, 4, 5):
        violations.append(f"invalid community card count: {len(curr['community_cards'])}")

    # Existing community cards shouldn't change
    prev_count = len(prev["community_cards"])
    if prev_count > 0 and len(curr["community_cards"]) >= prev_count:
        if prev["community_cards"] != curr["community_cards"][:prev_count]:
            violations.append(f"community cards changed: {prev['community_cards']} → {curr['community_cards'][:prev_count]}")

    # Pot can't decrease (within same hand)
    if prev["pot"] is not None and curr["pot"] is not None:
        if curr["pot"] < prev["pot"]:
            violations.append(f"pot decreased: {prev['pot']} → {curr['pot']}")

    # Player chips can't increase (no wins tracked)
    for p_prev, p_curr in zip(prev["players"], curr["players"]):
        if p_prev["chips"] is not None and p_curr["chips"] is not None:
            if p_curr["chips"] > p_prev["chips"]:
                violations.append(f"{p_curr['name']} chips increased: {p_prev['chips']} → {p_curr['chips']}")

    # Total chips in circulation must be conserved
    prev_total = _total_chips(prev)
    curr_total = _total_chips(curr)
    if prev_total is not None and curr_total is not None:
        if curr_total != prev_total:
            violations.append(f"total chips changed: {prev_total} → {curr_total}")

    return len(violations) == 0, violations


def analyze_video(
    video_path: str,
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    consensus_frames: int = CONSENSUS_FRAMES,
    include_rejected: bool = False,
) -> list[dict]:
    """
    Analyze a video file and extract game states.

    Args:
        video_path: Path to the video file.
        start_ms: Start timestamp in milliseconds.
        end_ms: End timestamp in milliseconds (None = entire video).
        consensus_frames: Number of consecutive valid frames needed to accept change.
        include_rejected: If True, include rejected states with rejected=True.

    Returns:
        List of state dicts with keys:
            - frame_number: int
            - timestamp_ms: float
            - community_cards: list[str]
            - pot: int or None
            - players: list[dict] with keys: name, chips
            - rejected: bool (only if include_rejected=True)
    """
    clear_caches()

    table = TableInfo.from_video(video_path)
    if not table.positions:
        return []

    states = []
    pending_buffer = []  # Buffer for valid frames proposing a state change

    with VideoFrameExtractor(video_path) as video:
        if end_ms is None:
            end_ms = video.duration_seconds * 1000

        start_frame = int(start_ms * video.fps / 1000)
        end_frame = int(end_ms * video.fps / 1000)
        total_frames = end_frame - start_frame

        for i, frame_info in enumerate(video.iterate_frames(start_frame, end_frame)):
            # Progress bar
            pct = (i + 1) / total_frames
            bar_width = 40
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r[{bar}] {pct*100:.0f}% ({i+1}/{total_frames})", end="", file=sys.stderr, flush=True)
            frame = frame_info.image

            # Extract all data from frame
            community = [c for c in match_community_cards(frame) if c]
            pot = extract_pot(frame)

            player_chips = []
            for i in range(len(table.names)):
                chips = extract_player_money(frame, table, i)
                player_chips.append(chips)

            state = {
                "frame_number": frame_info.frame_number,
                "timestamp_ms": frame_info.timestamp_ms,
                "community_cards": community,
                "pot": pot,
                "players": [{"name": name, "chips": chips}
                           for name, chips in zip(table.names, player_chips)],
            }

            # First state - just accept it
            if not states:
                if _is_complete(state):
                    states.append(state)
                elif include_rejected:
                    state["rejected"] = True
                    states.append(state)
                continue

            # Get last accepted state
            current = None
            for s in reversed(states):
                if not s.get("rejected"):
                    current = s
                    break

            if not current:
                states.append(state)
                continue

            # Check completeness
            if not _is_complete(state):
                if include_rejected:
                    state["rejected"] = True
                    states.append(state)
                pending_buffer = []
                continue

            # Validate transition
            is_valid, violations = _validate_transition(current, state)

            if not is_valid:
                if include_rejected:
                    state["rejected"] = True
                    state["violations"] = violations
                    states.append(state)
                pending_buffer = []
                continue

            # Valid frame - check if it's a change
            if _states_equivalent(current, state):
                # No change, reset pending buffer
                pending_buffer = []
                continue

            # Valid change - add to pending buffer
            if pending_buffer and _states_equivalent(pending_buffer[-1], state):
                pending_buffer.append(state)
            else:
                pending_buffer = [state]

            # Accept when we have enough consecutive valid frames
            if len(pending_buffer) >= consensus_frames:
                states.append(pending_buffer[-1])
                pending_buffer = []

        print(file=sys.stderr)  # Clear progress bar line

    return states


def analyze_and_print(
    video_path: str,
    start_s: float = 0,
    end_s: float | None = None,
    verbose: bool = False,
) -> None:
    """
    Analyze a video and print formatted output showing only changes.

    Args:
        video_path: Path to the video file.
        start_s: Start timestamp in seconds.
        end_s: End timestamp in seconds (None = entire video).
        verbose: If True, show rejected states with [X] prefix.
    """
    table = TableInfo.from_video(video_path)
    print(f"Hero: {' '.join(table.hero_cards)}")

    start_ms = start_s * 1000
    end_ms = end_s * 1000 if end_s else None
    states = analyze_video(video_path, start_ms, end_ms, include_rejected=verbose)

    if not states:
        print("No valid frames found.")
        return

    prev_state = None
    for state in states:
        is_rejected = state.get("rejected", False)

        # Skip rejected unless verbose
        if is_rejected and not verbose:
            continue

        parts = []

        # Board changes
        curr_board = state["community_cards"]
        prev_board = prev_state["community_cards"] if prev_state else []
        if curr_board != prev_board and curr_board:
            parts.append(f"Board: {' '.join(curr_board)}")

        # Pot changes
        curr_pot = state["pot"]
        prev_pot = prev_state["pot"] if prev_state else None
        if curr_pot != prev_pot and curr_pot:
            parts.append(f"Pot: {curr_pot:,}")

        # Player chip changes
        for i, p in enumerate(state["players"]):
            prev_chips = prev_state["players"][i]["chips"] if prev_state else None
            if p["chips"] != prev_chips and p["chips"]:
                parts.append(f"{p['name']}: {p['chips']:,}")

        if parts:
            prefix = "[X] " if is_rejected else ""
            print(f"{prefix}[{state['timestamp_ms']/1000:.1f}s] {' | '.join(parts)}")

        # Only update prev for non-rejected states
        if not is_rejected:
            prev_state = state
