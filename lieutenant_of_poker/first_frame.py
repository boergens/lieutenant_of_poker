"""
First-frame detection for Governor of Poker.

Detects static information from initial frames: active players, dealer button,
and hero cards. These don't change during a hand, so detection runs once.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .table_regions import NUM_PLAYERS, Region, detect_table_regions



@dataclass
class ActivePlayer:
    """Info about an active player."""
    name: Optional[str] = None
    is_hero: bool = False
    chip_region: Optional[Region] = None  # ROI for chip/money detection


@dataclass
class FirstFrameInfo:
    """Results from first-frame detection.

    Players are numbered 0..n-1 where the hero is always the last player.
    The physical seat layout is abstracted away - each player comes with
    their chip detection region so consumers don't need seat knowledge.
    """
    players: List[ActivePlayer] = field(default_factory=list)
    button_index: Optional[int] = None  # Index into players list
    hero_cards: List[str] = field(default_factory=list)  # Card strings like "Ah", "Kc"

    @property
    def num_players(self) -> int:
        """Number of active players."""
        return len(self.players)

    @property
    def hero_index(self) -> Optional[int]:
        """Index of the hero (always last player, or None if no players)."""
        return len(self.players) - 1 if self.players else None

    @property
    def player_names(self) -> List[str]:
        """Ordered list of player names."""
        return [p.name or f"Player{i}" for i, p in enumerate(self.players)]

    def __str__(self) -> str:
        """Format first frame info for display."""
        lines = []

        # Button position
        if self.button_index is not None:
            btn_player = self.players[self.button_index]
            btn_name = btn_player.name or f"Player {self.button_index}"
            lines.append(f"Button: {btn_name}")
        else:
            lines.append("Button: (not detected)")

        # Players with names
        detected = {i: p.name for i, p in enumerate(self.players) if p.name}
        if detected:
            lines.append(f"Players: {detected}")
        else:
            lines.append("Players: (none detected)")

        return "\n".join(lines)


def detect_first_frame(frame: np.ndarray) -> FirstFrameInfo:
    """
    Detect all static information from a single frame.

    Note: For reliable active player detection, use detect_first_frame_majority
    with multiple frames instead.

    Args:
        frame: BGR image frame from the game.

    Returns:
        FirstFrameInfo with detected values.
    """
    from .name_detector import detect_player_names
    from .dealer_detector import detect_dealer_position
    from .card_matcher import match_hero_cards
    from .chip_ocr import extract_player_chips

    region_detector = detect_table_regions(frame)

    # Detect names for all positions
    names = detect_player_names(frame, scale_frame=False)

    # Detect active players (seats where chips can be read)
    # Collect as (seat, player) tuples, then sort by seat so hero (highest) is last
    active_seats = []
    for seat in range(NUM_PLAYERS):
        chips = extract_player_chips(frame, region_detector, seat)
        if chips is not None:
            active_seats.append(seat)

    # Sort by seat number (hero at highest seat will be last)
    active_seats.sort()
    hero_seat = active_seats[-1] if active_seats else None

    # Build players list with chip regions
    players = []
    seat_to_index = {}
    for idx, seat in enumerate(active_seats):
        chip_region = region_detector.get_player_chip_region(seat)
        players.append(ActivePlayer(
            name=names.get(seat),
            is_hero=(seat == hero_seat),
            chip_region=chip_region,
        ))
        seat_to_index[seat] = idx

    # Detect button and convert to player index
    button_seat = detect_dealer_position(frame, region_detector)
    button_index = seat_to_index.get(button_seat) if button_seat is not None else None

    # Detect hero cards directly from frame
    hero_cards = [c for c in match_hero_cards(frame) if c]

    return FirstFrameInfo(
        players=players,
        button_index=button_index,
        hero_cards=hero_cards,
    )


def detect_first_frame_majority(frames: List[np.ndarray], min_valid_frames: int = 3) -> FirstFrameInfo:
    """
    Detect static information using majority voting across multiple frames.

    A seat is considered active if chips could be read in at least min_valid_frames.
    The actual chip values don't need to match - just that valid numbers were detected.

    Args:
        frames: List of BGR image frames (typically 3+).
        min_valid_frames: Minimum frames with valid chip reading to consider seat active.

    Returns:
        FirstFrameInfo with active players and majority-voted button position.
    """
    from .name_detector import detect_player_names
    from .dealer_detector import detect_dealer_position
    from .card_matcher import match_hero_cards
    from .chip_ocr import extract_player_chips

    if not frames:
        return FirstFrameInfo()

    # Use first frame for names, hero cards, and region detector
    first = frames[0]
    region_detector = detect_table_regions(first)
    names = detect_player_names(first, scale_frame=False)
    hero_cards = [c for c in match_hero_cards(first) if c]

    # Count valid chip readings per seat across all frames
    valid_chip_counts: Dict[int, int] = {seat: 0 for seat in range(NUM_PLAYERS)}

    button_votes = []
    for frame in frames:
        rd = detect_table_regions(frame)

        # Check chip readability for each seat
        for seat in range(NUM_PLAYERS):
            chips = extract_player_chips(frame, rd, seat)
            if chips is not None:
                valid_chip_counts[seat] += 1

        # Collect button position votes
        pos = detect_dealer_position(frame, rd)
        if pos is not None:
            button_votes.append(pos)

    # Determine active seats (those with enough valid chip readings)
    active_seats = [seat for seat in range(NUM_PLAYERS)
                    if valid_chip_counts[seat] >= min_valid_frames]
    active_seats.sort()  # Sort so hero (highest seat) is last
    hero_seat = active_seats[-1] if active_seats else None

    # Build players list with chip regions
    players = []
    seat_to_index = {}
    for idx, seat in enumerate(active_seats):
        chip_region = region_detector.get_player_chip_region(seat)
        players.append(ActivePlayer(
            name=names.get(seat),
            is_hero=(seat == hero_seat),
            chip_region=chip_region,
        ))
        seat_to_index[seat] = idx

    # Majority vote for button position, convert to player index
    button_index = None
    if button_votes:
        counter = Counter(button_votes)
        button_seat = counter.most_common(1)[0][0]
        button_index = seat_to_index.get(button_seat)

    return FirstFrameInfo(
        players=players,
        button_index=button_index,
        hero_cards=hero_cards,
    )


def detect_from_video(video_path: str, start_ms: float = 0, num_frames: int = 3) -> FirstFrameInfo:
    """
    Detect static information from a video file using majority voting.

    Grabs multiple consecutive frames and uses majority voting for active
    player detection and button position.

    Args:
        video_path: Path to the video file.
        start_ms: Starting timestamp in milliseconds.
        num_frames: Number of frames to sample for voting (default 3).

    Returns:
        FirstFrameInfo with detected values.
    """
    from .frame_extractor import VideoFrameExtractor, get_video_info

    info = get_video_info(video_path)
    frame_interval = 1000 / info['fps']

    frames = []
    with VideoFrameExtractor(video_path) as extractor:
        for i in range(num_frames):
            ts = start_ms + i * frame_interval
            frame_info = extractor.get_frame_at_timestamp(ts)
            if frame_info is not None:
                frames.append(frame_info.image)

    if not frames:
        return FirstFrameInfo()

    return detect_first_frame_majority(frames, min_valid_frames=min(3, len(frames)))
