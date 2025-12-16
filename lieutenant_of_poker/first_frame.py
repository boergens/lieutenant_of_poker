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

# Load active seat templates (card back pattern shown for non-hero active players)
_ASSETS_DIR = Path(__file__).parent / "assets"
_ACTIVE_SEAT_LEFT = cv2.imread(str(_ASSETS_DIR / "active_seat_left.png"))
_ACTIVE_SEAT_RIGHT = cv2.imread(str(_ASSETS_DIR / "active_seat_right.png"))


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


def _get_template_score(frame: np.ndarray, pos: Tuple[int, int]) -> float:
    """Get best template match score for a blind indicator at position."""
    if not _BLIND_TEMPLATES:
        return 0.0

    px, py = pos
    height, width = frame.shape[:2]

    if px < 0 or py < 0 or px + 10 > width or py + 10 > height:
        return 0.0

    region = frame[py:py + 10, px:px + 10]
    if region.shape != (10, 10, 3):
        return 0.0

    best_score = 0.0
    for template in _BLIND_TEMPLATES:
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0]
        if score > best_score:
            best_score = score
    return best_score


def detect_blinds(frame: np.ndarray, seat_indices: List[int]) -> tuple[bool, List[dict]]:
    """
    Detect blinds at specified seat positions.

    Args:
        frame: BGR game frame.
        seat_indices: List of seat indices to check.

    Returns:
        Tuple of (no_currency, results) where results is a list of dicts with:
        seat_index, position, region, template_score, is_empty_felt, has_indicator, amount
    """
    from .chip_ocr import extract_money_at

    # First pass: check if any blind indicator found (determines currency mode)
    any_indicator = False
    for seat_idx in seat_indices:
        blind_pos = _BLIND_POSITIONS[seat_idx]
        if blind_pos is not None and _has_blind_indicator(frame, blind_pos):
            any_indicator = True
            break

    no_currency = not any_indicator

    results = []
    for seat_idx in seat_indices:
        blind_pos = _BLIND_POSITIONS[seat_idx]

        if blind_pos is None:
            results.append({
                "seat_index": seat_idx,
                "position": None,
                "region": None,
                "template_score": 0.0,
                "is_empty_felt": True,
                "has_indicator": False,
                "amount": None,
            })
            continue

        px, py = blind_pos
        height, width = frame.shape[:2]

        # Extract 10x10 region
        region = None
        if 0 <= px and 0 <= py and px + 10 <= width and py + 10 <= height:
            region = frame[py:py + 10, px:px + 10]

        template_score = _get_template_score(frame, blind_pos)
        is_empty = _is_empty_felt(frame, blind_pos)
        has_indicator = _has_blind_indicator(frame, blind_pos)

        # Extract amount based on currency mode
        amount = None
        if no_currency:
            if not is_empty:
                amount = extract_money_at(frame, blind_pos, no_currency=True)
        elif has_indicator:
            amount = extract_money_at(frame, blind_pos, no_currency=False)

        results.append({
            "seat_index": seat_idx,
            "position": blind_pos,
            "region": region,
            "template_score": template_score,
            "is_empty_felt": is_empty,
            "has_indicator": has_indicator,
            "amount": amount,
        })

    return no_currency, results


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

        # Detect active seats using card pattern matching
        from .card_matcher import is_seat_active
        first_frame = frames[0]
        active_seats = [i for i, pos in enumerate(_SEAT_POSITIONS) if is_seat_active(first_frame, pos)]

        # Collect name votes for active positions
        position_votes = {i: [] for i in active_seats}
        button_votes = []
        hero_cards = ()

        for frame in frames:
            # OCR each active position
            for i in active_seats:
                pos_x, pos_y = _SEAT_POSITIONS[i]
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

        # Build player list from active seats
        names = []
        positions = []
        seat_indices = []
        for i in active_seats:
            votes = position_votes[i]
            if votes:
                name = Counter(votes).most_common(1)[0][0]
            else:
                name = f"Player_{i}"
            names.append(name)
            positions.append(_SEAT_POSITIONS[i])
            seat_indices.append(i)

        # Button: find closest seat (may be inactive), then adjust
        button_player_idx = None
        if button_votes and active_seats:
            button_pos = Counter(button_votes).most_common(1)[0][0]
            bx, by = button_pos

            # Find closest seat (including inactive ones)
            min_dist = float('inf')
            closest_seat = 0
            for seat_idx, (px, py) in enumerate(_SEAT_POSITIONS):
                dist = (px - bx) ** 2 + (py - by) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_seat = seat_idx

            # If closest seat is inactive, move button to previous active seat
            # (counter-clockwise = decreasing seat index)
            # The button would have gone to this player if the inactive seat wasn't there
            if closest_seat not in active_seats:
                for offset in range(1, len(_SEAT_POSITIONS)):
                    prev_seat = (closest_seat - offset) % len(_SEAT_POSITIONS)
                    if prev_seat in active_seats:
                        closest_seat = prev_seat
                        break

            # Convert seat index to player index (before rotation)
            if closest_seat in seat_indices:
                button_player_idx = seat_indices.index(closest_seat)

        # Detect blinds using first frame
        first_frame = frames[0]
        no_currency, blind_results = detect_blinds(first_frame, seat_indices)
        blind_amounts_by_seat = [r["amount"] for r in blind_results]

        # Find hero and rotate
        if names:
            hero_idx = max(
                range(len(names)),
                key=lambda i: SequenceMatcher(None, names[i].lower(), cls._HERO_NAME.lower()).ratio()
            )
            names = names[hero_idx + 1:] + names[:hero_idx + 1]
            positions = positions[hero_idx + 1:] + positions[:hero_idx + 1]
            blind_amounts_by_seat = blind_amounts_by_seat[hero_idx + 1:] + blind_amounts_by_seat[:hero_idx + 1]
            # Rotate button index too
            if button_player_idx is not None:
                button_player_idx = (button_player_idx - hero_idx - 1) % len(names)

        # Now detect hero cards using hero's position
        hero_cards = ()
        if positions:
            hero_pos = positions[-1]  # Hero is last
            for frame in frames:
                hero_cards = tuple(c for c in match_hero_cards(frame, hero_pos) if c)
                if hero_cards:
                    break

        button_index = button_player_idx

        return cls(
            names=tuple(names),
            positions=tuple(positions),
            button_index=button_index,
            hero_cards=hero_cards,
            blind_amounts=tuple(blind_amounts_by_seat),
            no_currency=no_currency,
        )
