"""
First-frame detection for Governor of Poker.

Detects static information from initial frames: active players, dealer button,
and hero cards. These don't change during a hand, so detection runs once.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ._positions import SEAT_POSITIONS as _SEAT_POSITIONS


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

    def __str__(self) -> str:
        lines = [f"Players: {list(self.names)}"]
        lines.append(f"Button: {self.button_index}")
        lines.append(f"Hero cards: {self.hero_cards}")
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
            return cls(names=(), positions=(), button_index=None, hero_cards=())

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
        names = []
        positions = []
        for i, votes in position_votes.items():
            if len(votes) >= 2:
                # Most common name for this position
                name = Counter(votes).most_common(1)[0][0]
                names.append(name)
                positions.append(_SEAT_POSITIONS[i])

        # Find hero and rotate
        if names:
            hero_idx = max(
                range(len(names)),
                key=lambda i: SequenceMatcher(None, names[i].lower(), cls._HERO_NAME.lower()).ratio()
            )
            names = names[hero_idx + 1:] + names[:hero_idx + 1]
            positions = positions[hero_idx + 1:] + positions[:hero_idx + 1]

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
        )
