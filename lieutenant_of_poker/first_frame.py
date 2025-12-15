"""
First-frame detection for Governor of Poker.

Detects static information from initial frames: active players, dealer button,
and hero cards. These don't change during a hand, so detection runs once.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ._positions import SEAT_POSITIONS as _SEAT_POSITIONS
from ._positions import BLIND_POSITIONS as _BLIND_POSITIONS


# Load blind indicator templates on module import
_BLIND_TEMPLATES: List[np.ndarray] = []
_TEMPLATES_DIR = Path(__file__).parent / "assets" / "blind_templates"
for _template_path in sorted(_TEMPLATES_DIR.glob("blind_debug_seat*.png")):
    _template = cv2.imread(str(_template_path))
    if _template is not None:
        _BLIND_TEMPLATES.append(_template)


def _has_blind_indicator(frame: np.ndarray, pos: Tuple[int, int], threshold: float = 0.8) -> bool:
    """
    Check if a blind indicator is present at the given position.

    Compares a 10x10 region at pos against known blind indicator templates.

    Args:
        frame: BGR game frame.
        pos: (x, y) coordinates of the blind indicator position.
        threshold: Minimum match score (0-1) to consider a match.

    Returns:
        True if a blind indicator is detected, False otherwise.
    """
    if not _BLIND_TEMPLATES:
        return True  # No templates loaded, assume blind present

    px, py = pos
    height, width = frame.shape[:2]

    # Bounds check
    if px < 0 or py < 0 or px + 10 > width or py + 10 > height:
        return False

    region = frame[py:py + 10, px:px + 10]

    if region.shape != (10, 10, 3):
        return False

    # Check against each template
    for template in _BLIND_TEMPLATES:
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0]
        if score >= threshold:
            return True

    return False


def _is_empty_felt(frame: np.ndarray, pos: Tuple[int, int], region_size: int = 20) -> bool:
    """
    Check if a region is empty poker felt (uniform green).

    Args:
        frame: BGR game frame.
        pos: (x, y) coordinates to check.
        region_size: Size of region to check.

    Returns:
        True if region is uniform green (empty felt), False if it has content.
    """
    px, py = pos
    height, width = frame.shape[:2]

    # Bounds check
    if px < 0 or py < 0 or px + region_size > width or py + region_size > height:
        return True  # Out of bounds, treat as empty

    region = frame[py:py + region_size, px:px + region_size]

    # Check color variance - empty felt should be very uniform
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Check if mostly green (hue around 60-90 for green felt)
    h, s, v = cv2.split(hsv)
    mean_h = np.mean(h)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    # Empty felt: green hue, low variance across all channels
    is_green = 30 < mean_h < 80
    is_uniform = std_h < 10 and std_s < 20 and std_v < 20

    return is_green and is_uniform


@dataclass(frozen=True)
class TableInfo:
    """
    Immutable game table information detected from first frame(s).

    Players are ordered with hero always last.
    """
    names: Tuple[str, ...]
    positions: Tuple[Tuple[int, int], ...]
    button_index: Optional[int]
    hero_cards: Tuple[str, ...]
    # Blind amounts detected at each seat (None if no blind at that seat)
    # Indexed by position in the names/positions tuples
    blind_amounts: Tuple[Optional[int], ...]
    # True if no currency symbol was detected (money boxes shifted left)
    no_currency: bool

    def __str__(self) -> str:
        lines = [f"Players: {list(self.names)}"]
        lines.append(f"Button: {self.button_index}")
        lines.append(f"Hero cards: {self.hero_cards}")
        # Show detected blinds
        blinds = [(i, b) for i, b in enumerate(self.blind_amounts) if b is not None]
        if blinds:
            blind_strs = [f"{self.names[i]}: {b}" for i, b in blinds]
            lines.append(f"Blinds: {', '.join(blind_strs)}")
        lines.append(f"Currency: {'none' if self.no_currency else 'detected'}")
        return "\n".join(lines)

    # Class constants (internal)
    _NAME_WIDTH: int = 140
    _NAME_HEIGHT: int = 30
    _HERO_NAME: str = "kevinLAS"

    @classmethod
    def from_video(cls, video_path: str) -> "TableInfo":
        """Detect table info from a video file using majority vote across 3 frames."""
        from collections import Counter
        from .frame_extractor import VideoFrameExtractor
        from .card_matcher import match_hero_cards
        from .fast_ocr import ocr_name
        from .chip_ocr import extract_money_at

        # Load dealer button template
        dealer_template = cv2.imread(str(Path(__file__).parent / "dealer.png"))

        # Sample 3 frames, 1 second apart
        frames = []
        with VideoFrameExtractor(video_path) as extractor:
            for t in (0, 1000, 2000):
                frame_info = extractor.get_frame_at_timestamp(t)
                if frame_info is not None:
                    frames.append(frame_info.image)

        if not frames:
            raise ValueError(f"Could not extract any frames from video: {video_path}")

        # Collect votes for each position
        position_votes = {i: [] for i in range(len(_SEAT_POSITIONS))}
        button_votes = []
        hero_cards = ()

        for frame in frames:
            # OCR each position
            for i, (pos_x, pos_y) in enumerate(_SEAT_POSITIONS):
                x = pos_x
                y = pos_y - cls._NAME_HEIGHT
                region = frame[y:y + cls._NAME_HEIGHT, x:x + cls._NAME_WIDTH]
                name = ocr_name(region)
                if name is not None:
                    position_votes[i].append(name)

            # Find dealer button via template matching
            if dealer_template is not None:
                result = cv2.matchTemplate(frame, dealer_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.7:
                    th, tw = dealer_template.shape[:2]
                    button_center = (max_loc[0] + tw // 2, max_loc[1] + th // 2)
                    button_votes.append(button_center)

        # Majority vote: position is active if detected in 2+ frames
        # Track seat indices for blind position mapping
        names = []
        positions = []
        seat_indices = []  # Original seat index for each active player
        for i, votes in position_votes.items():
            if len(votes) >= 2:
                # Most common name for this position
                name = Counter(votes).most_common(1)[0][0]
                names.append(name)
                positions.append(_SEAT_POSITIONS[i])
                seat_indices.append(i)

        # Check if any blind indicator is found at known positions
        # If none found, switch to no-currency mode (shift money boxes left)
        first_frame = frames[0]
        any_blind_found = False
        for seat_idx in seat_indices:
            blind_pos = _BLIND_POSITIONS[seat_idx]
            if blind_pos is not None and _has_blind_indicator(first_frame, blind_pos):
                any_blind_found = True
                break
        no_currency = not any_blind_found

        # Extract blind amounts from first frame only (no consensus)
        # Currency mode: only extract if blind indicator template matches
        # No-currency mode: extract if region is not empty felt
        blind_amounts_by_seat = []
        for seat_idx in seat_indices:
            blind_pos = _BLIND_POSITIONS[seat_idx]
            if blind_pos is None:
                blind_amounts_by_seat.append(None)
            elif no_currency:
                # No currency: check if region has content (not empty felt)
                if not _is_empty_felt(first_frame, blind_pos):
                    amount = extract_money_at(first_frame, blind_pos, no_currency=True)
                    blind_amounts_by_seat.append(amount)
                else:
                    blind_amounts_by_seat.append(None)
            elif _has_blind_indicator(first_frame, blind_pos):
                # Currency mode: use template matching
                amount = extract_money_at(first_frame, blind_pos, no_currency=False)
                blind_amounts_by_seat.append(amount)
            else:
                blind_amounts_by_seat.append(None)

        # Find hero and rotate
        if names:
            hero_idx = max(
                range(len(names)),
                key=lambda i: SequenceMatcher(None, names[i].lower(), cls._HERO_NAME.lower()).ratio()
            )
            names = names[hero_idx + 1:] + names[:hero_idx + 1]
            positions = positions[hero_idx + 1:] + positions[:hero_idx + 1]
            blind_amounts_by_seat = blind_amounts_by_seat[hero_idx + 1:] + blind_amounts_by_seat[:hero_idx + 1]

        # Now detect hero cards using hero's position
        hero_cards = ()
        if positions:
            hero_pos = positions[-1]  # Hero is last
            for frame in frames:
                hero_cards = tuple(c for c in match_hero_cards(frame, hero_pos) if c)
                if hero_cards:
                    break

        # Button: majority vote, then find closest player
        button_index = None
        if button_votes and positions:
            button_pos = Counter(button_votes).most_common(1)[0][0]
            min_dist = float('inf')
            for idx, (px, py) in enumerate(positions):
                bx, by = button_pos
                dist = (px - bx) ** 2 + (py - by) ** 2
                if dist < min_dist:
                    min_dist = dist
                    button_index = idx

        return cls(
            names=tuple(names),
            positions=tuple(positions),
            button_index=button_index,
            hero_cards=hero_cards,
            blind_amounts=tuple(blind_amounts_by_seat),
            no_currency=no_currency,
        )
