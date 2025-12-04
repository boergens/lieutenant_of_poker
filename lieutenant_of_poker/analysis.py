"""
High-level analysis operations for video processing.

This module provides the main entry points for analyzing poker videos,
extracting frames, and generating reports.
"""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional, Callable, List, TypeVar

from .frame_extractor import VideoFrameExtractor
from .game_state import GameStateExtractor, GameState, PlayerState
from .first_frame import FirstFrameInfo, ActivePlayer, detect_first_frame_majority

T = TypeVar("T")

# Number of consecutive matching rejected frames needed to accept a state change
CONSENSUS_FRAMES = 3


def states_equivalent(state1: GameState, state2: GameState) -> bool:
    """
    Check if two game states have equivalent significant values.

    Compares hero cards, community cards, pot, hero chips, and player chips.
    Used to detect if consecutive rejected frames are requesting the same change.
    """
    # Compare hero cards
    if len(state1.hero_cards) != len(state2.hero_cards):
        return False
    for c1, c2 in zip(state1.hero_cards, state2.hero_cards):
        if c1.rank != c2.rank or c1.suit != c2.suit:
            return False

    # Compare community cards
    if len(state1.community_cards) != len(state2.community_cards):
        return False
    for c1, c2 in zip(state1.community_cards, state2.community_cards):
        if c1.rank != c2.rank or c1.suit != c2.suit:
            return False

    # Compare pot
    if state1.pot != state2.pot:
        return False

    # Compare hero chips
    if state1.hero_chips != state2.hero_chips:
        return False

    # Compare player chips
    if set(state1.players.keys()) != set(state2.players.keys()):
        return False
    for pos in state1.players:
        if state1.players[pos].chips != state2.players[pos].chips:
            return False

    return True


def majority_vote(values: List[T]) -> Optional[T]:
    """
    Return the most common value from a list, or None if the list is empty.

    For values that appear equally often, returns the first one encountered.
    None values are filtered out before voting.
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    counter = Counter(valid_values)
    return counter.most_common(1)[0][0]


def compute_initial_state(states: List[GameState]) -> GameState:
    """
    Compute a consolidated initial state from multiple frame states using majority voting.

    Pools hero cards and chip values from the provided states and returns
    a new GameState with the majority-voted values.

    Args:
        states: List of GameState objects from the first few frames.

    Returns:
        A consolidated GameState with majority-voted values.
    """
    if not states:
        return GameState()

    # Use the first state as the base (for metadata like frame_number, timestamp)
    base_state = states[0]

    # Majority vote for hero cards
    # Cards are compared by their (rank, suit) tuple for hashability
    hero_cards_left = []
    hero_cards_right = []
    for state in states:
        if len(state.hero_cards) >= 1:
            hero_cards_left.append((state.hero_cards[0].rank, state.hero_cards[0].suit))
        if len(state.hero_cards) >= 2:
            hero_cards_right.append((state.hero_cards[1].rank, state.hero_cards[1].suit))

    from .card_detector import Card
    voted_hero_cards = []
    voted_left = majority_vote(hero_cards_left)
    if voted_left:
        voted_hero_cards.append(Card(rank=voted_left[0], suit=voted_left[1]))
    voted_right = majority_vote(hero_cards_right)
    if voted_right:
        voted_hero_cards.append(Card(rank=voted_right[0], suit=voted_right[1]))

    # Majority vote for pot
    pots = [s.pot for s in states]
    voted_pot = majority_vote(pots)

    # Majority vote for hero_chips
    hero_chips_list = [s.hero_chips for s in states]
    voted_hero_chips = majority_vote(hero_chips_list)

    # Majority vote for each player's chips
    # First, collect all player positions that appear
    all_positions = set()
    for state in states:
        all_positions.update(state.players.keys())

    voted_players = {}
    for pos in all_positions:
        chips_list = [s.players.get(pos, PlayerState(position=pos)).chips for s in states]
        voted_chips = majority_vote(chips_list)

        # Get a base PlayerState from the first state that has this position
        base_player = None
        for state in states:
            if pos in state.players:
                base_player = state.players[pos]
                break

        if base_player:
            voted_players[pos] = PlayerState(
                position=pos,
                name=base_player.name,
                chips=voted_chips,
                cards=base_player.cards,
                last_action=base_player.last_action,
                is_active=base_player.is_active,
                is_dealer=base_player.is_dealer,
            )

    return GameState(
        community_cards=[],  # No community cards at start
        hero_cards=voted_hero_cards,
        pot=voted_pot,
        hero_chips=voted_hero_chips,
        players=voted_players,
        street=base_state.street,
        frame_number=base_state.frame_number,
        timestamp_ms=base_state.timestamp_ms,
    )


@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""

    start_ms: float = 0
    end_ms: Optional[float] = None
    table_background: Optional[str] = None  # Path to table background image


@dataclass
class AnalysisProgress:
    """Progress information during analysis."""

    current_frame: int
    total_frames: int
    timestamp_ms: float
    ocr_calls: int = 0


def analyze_video(
    video_path: str,
    config: AnalysisConfig,
    on_progress: Optional[Callable[[AnalysisProgress], None]] = None,
    initial_frames: int = 3,
    consensus_frames: int = CONSENSUS_FRAMES,
    include_rejected: bool = False,
) -> List[GameState]:
    """
    Analyze a video file and extract game states.

    Assumes we're analyzing a single poker hand from a clean starting state
    (hero cards dealt, blinds posted, no community cards).

    State changes are only accepted when 3 consecutive VALID frames agree
    on the same change. Invalid frames (OCR errors, impossible states) are
    discarded completely.

    Args:
        video_path: Path to the video file.
        config: Analysis configuration.
        on_progress: Optional callback for progress updates.
        initial_frames: Number of frames to pool for initial state (default 3).
        consensus_frames: Number of consecutive VALID frames needed to accept
                         a state change (default 3).
        include_rejected: If True, include rejected states in output with
                         rejected=True flag set. Default False.

    Returns:
        List of GameState objects extracted from the video.
    """
    from .chip_ocr import get_ocr_calls, clear_caches
    from .fast_ocr import set_ocr_debug_context
    from .rules_validator import is_complete_frame, validate_transition

    extractor = GameStateExtractor(table_background=config.table_background)
    clear_caches()

    states = []
    initial_states = []  # Collect first N frames for majority voting
    initial_images = []  # Collect frame images for first_frame detection
    pending_change_buffer = []  # Buffer for valid frames proposing a state change
    players: Optional[List[ActivePlayer]] = None  # Set after first_frame detection

    with VideoFrameExtractor(video_path) as video:
        start_ms = config.start_ms
        end_ms = config.end_ms if config.end_ms else video.duration_seconds * 1000

        # Calculate frame range from timestamps
        start_frame = int(start_ms * video.fps / 1000)
        end_frame = int(end_ms * video.fps / 1000) if end_ms else video.frame_count

        total_frames = end_frame - start_frame
        current_frame = 0

        for frame_info in video.iterate_frames(start_frame, end_frame):
            # Set OCR debug context for this frame
            set_ocr_debug_context(video_path, frame_info.timestamp_ms)

            # Pool first N frames for first_frame detection and initial state
            if current_frame < initial_frames:
                initial_images.append(frame_info.image)

                # When we have enough frames, detect active players and compute initial state
                if len(initial_images) == initial_frames:
                    # Detect active players from initial frames
                    first_frame_info = detect_first_frame_majority(initial_images)
                    players = first_frame_info.players

                    # Now extract states with players known
                    for i, img in enumerate(initial_images):
                        state = extractor.extract(
                            img,
                            frame_number=start_frame + i,
                            timestamp_ms=start_ms + i * (1000 / video.fps),
                            players=players,
                        )
                        initial_states.append(state)

                    consolidated = compute_initial_state(initial_states)
                    states.append(consolidated)
            else:
                # Extract state for this frame with players
                state = extractor.extract(
                    frame_info.image,
                    frame_number=frame_info.frame_number,
                    timestamp_ms=frame_info.timestamp_ms,
                    players=players,
                )

                # First check: frame must have all values (no None in critical fields)
                if not is_complete_frame(state):
                    # Invalid frame - discard completely, reset pending buffer
                    if include_rejected:
                        state.rejected = True
                        states.append(state)
                    pending_change_buffer = []
                    current_frame += 1
                    if on_progress:
                        on_progress(AnalysisProgress(
                            current_frame=current_frame,
                            total_frames=total_frames,
                            timestamp_ms=frame_info.timestamp_ms,
                            ocr_calls=get_ocr_calls(),
                        ))
                    continue

                # Second check: validate against current accepted state
                # Find last accepted (non-rejected) state for comparison
                current_state = None
                for s in reversed(states):
                    if not s.rejected:
                        current_state = s
                        break
                if current_state:
                    result = validate_transition(
                        current_state, state,
                        allow_new_hand=False,
                        check_chip_increases=True,
                    )
                    if not result.is_valid:
                        # Invalid transition - discard frame, reset pending buffer
                        if include_rejected:
                            state.rejected = True
                            states.append(state)
                        pending_change_buffer = []
                    elif states_equivalent(current_state, state):
                        # Valid but no change - keep current state, reset pending buffer
                        pending_change_buffer = []
                    else:
                        # Valid frame with a state change - add to pending buffer
                        if pending_change_buffer and states_equivalent(pending_change_buffer[-1], state):
                            # Matches previous pending change
                            pending_change_buffer.append(state)
                        else:
                            # Different change - start new pending buffer
                            pending_change_buffer = [state]

                        # Accept change when we have enough consecutive valid frames
                        if len(pending_change_buffer) >= consensus_frames:
                            states.append(pending_change_buffer[-1])
                            pending_change_buffer = []
                else:
                    states.append(state)

            current_frame += 1

            if on_progress:
                on_progress(
                    AnalysisProgress(
                        current_frame=current_frame,
                        total_frames=total_frames,
                        timestamp_ms=frame_info.timestamp_ms,
                        ocr_calls=get_ocr_calls(),
                    )
                )

        # Handle edge case: video has fewer frames than initial_frames
        if initial_images and not states:
            # Detect active players from what we have
            first_frame_info = detect_first_frame_majority(
                initial_images, min_valid_frames=min(3, len(initial_images))
            )
            players = first_frame_info.players

            # Extract states with players
            for i, img in enumerate(initial_images):
                state = extractor.extract(
                    img,
                    frame_number=start_frame + i,
                    timestamp_ms=start_ms + i * (1000 / video.fps),
                    players=players,
                )
                initial_states.append(state)

            consolidated = compute_initial_state(initial_states)
            states.append(consolidated)

    return states
