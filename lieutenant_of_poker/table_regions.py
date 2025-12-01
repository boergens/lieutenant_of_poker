"""
Table region detection for Governor of Poker.

Defines the key UI regions of the poker table and provides utilities
for extracting sub-images from frames.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np


class PlayerPosition(Enum):
    """Player seat positions around the table (5 players, in playing order)."""
    SEAT_1 = auto()  # First opponent (left side)
    SEAT_2 = auto()  # Second opponent (top left)
    SEAT_3 = auto()  # Third opponent (top right)
    SEAT_4 = auto()  # Fourth opponent (right side)
    HERO = auto()    # Main player (bottom)


@dataclass
class Region:
    """A rectangular region in the frame."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the region."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract this region from a frame."""
        return frame[self.y:self.y2, self.x:self.x2].copy()

    def scale(self, scale_x: float, scale_y: float) -> "Region":
        """Return a new Region scaled by the given factors."""
        return Region(
            x=int(self.x * scale_x),
            y=int(self.y * scale_y),
            width=int(self.width * scale_x),
            height=int(self.height * scale_y)
        )

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this region."""
        return self.x <= x < self.x2 and self.y <= y < self.y2


@dataclass
class PlayerRegions:
    """Regions associated with a player position."""
    position: PlayerPosition
    name_chip_box: Region      # Player name and chip count area
    cards: Optional[Region]    # Hole cards (only visible for hero or showdown)
    action_label: Region       # Where action labels appear (CHECK, FOLD, etc.)


# Base resolution the regions are defined for (from sample frames at 50% scale)
BASE_WIDTH = 1728
BASE_HEIGHT = 1117


class TableRegionDetector:
    """
    Detects and provides access to key regions of the poker table UI.

    Regions are defined for a base resolution and scaled to match the actual
    frame size.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize the detector for a specific frame size.

        Args:
            frame_width: Width of video frames.
            frame_height: Height of video frames.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale_x = frame_width / BASE_WIDTH
        self.scale_y = frame_height / BASE_HEIGHT

        # Define base regions (at BASE_WIDTH x BASE_HEIGHT resolution)
        self._define_regions()

    def _define_regions(self) -> None:
        """Define all table regions at base resolution, then scale."""
        # Pot display region (the "1,120" text in center-right of table)
        self._pot_region = self._scaled(Region(x=765, y=356, width=204, height=64))

        # Community cards region (Q♥ 3♥ 3♦ area - below pot)
        self._community_cards_region = self._scaled(Region(x=560, y=420, width=430, height=140))

        # Individual community card slots (5 fixed positions)
        # Tuned for 10px background padding around each card
        card_width = 93  # Gives ~103px at output for 83px card + 20px padding
        card_spacing = -4  # 89px slot-to-slot spacing (was -7, cards drifted right)
        card_start_x = 641  # Shifted to center card with 10px left padding
        card_y = 429  # Moved down so card has 10px top padding
        card_height = 132  # Gives ~147px at output for 127px card + 20px padding
        self._community_card_slots = [
            self._scaled(Region(
                x=card_start_x + i * (card_width + card_spacing),
                y=card_y,
                width=card_width,
                height=card_height
            ))
            for i in range(5)
        ]

        # Hero's hole cards (shifted so previous center is now top-left)
        self._hero_cards_region = self._scaled(Region(x=740, y=655, width=210, height=110))

        # Individual hero card slots (2 fixed positions)
        hero_card_width = 100
        hero_card_spacing = 10
        hero_card_start_x = 740  # Shifted right by 50 (half width)
        hero_card_y = 655        # Shifted down by 55 (half height)
        hero_card_height = 110
        self._hero_card_slots = [
            self._scaled(Region(
                x=hero_card_start_x + i * (hero_card_width + hero_card_spacing),
                y=hero_card_y,
                width=hero_card_width,
                height=hero_card_height
            ))
            for i in range(2)
        ]

        # Action buttons region (CHECK/FOLD, CALL ANY, etc. at bottom)
        self._action_buttons_region = self._scaled(Region(x=380, y=780, width=550, height=50))

        # Player regions by position (corrected to BASE_WIDTH x BASE_HEIGHT = 1728x1117)
        self._player_regions = {
            PlayerPosition.SEAT_1: PlayerRegions(
                position=PlayerPosition.SEAT_1,
                # Chip: top half, trimmed left 25% / right 10%
                name_chip_box=self._scaled(Region(x=287, y=646, width=150, height=29)),
                cards=None,
                # Action: left half only
                action_label=self._scaled(Region(x=222, y=679, width=120, height=34)),
            ),
            PlayerPosition.SEAT_2: PlayerRegions(
                position=PlayerPosition.SEAT_2,
                # Chip: top half, trimmed left 25% / right 10%
                name_chip_box=self._scaled(Region(x=439, y=245, width=144, height=31)),
                cards=None,
                # Action: left half only
                action_label=self._scaled(Region(x=376, y=280, width=115, height=36)),
            ),
            PlayerPosition.SEAT_3: PlayerRegions(
                position=PlayerPosition.SEAT_3,
                # Chip: top half, trimmed left 25% / right 10%
                name_chip_box=self._scaled(Region(x=1179, y=251, width=147, height=26)),
                cards=None,
                # Action: left half only
                action_label=self._scaled(Region(x=1115, y=281, width=117, height=32)),
            ),
            PlayerPosition.SEAT_4: PlayerRegions(
                position=PlayerPosition.SEAT_4,
                # Chip: top half, trimmed left 25% / right 10%
                name_chip_box=self._scaled(Region(x=1341, y=645, width=143, height=28)),
                cards=None,
                # Action: left half only
                action_label=self._scaled(Region(x=1279, y=674, width=114, height=31)),
            ),
            PlayerPosition.HERO: PlayerRegions(
                position=PlayerPosition.HERO,
                name_chip_box=self._scaled(Region(x=974, y=845, width=128, height=29)),
                cards=self._hero_cards_region,
                action_label=self._scaled(Region(x=917, y=876, width=90, height=33)),
            ),
        }

        # Dealer button search region (covers table area where button can appear)
        self._dealer_button_search_region = self._scaled(Region(x=300, y=400, width=700, height=250))

        # Total chips/balance display (top left)
        self._balance_region = self._scaled(Region(x=200, y=80, width=120, height=40))

    def _scaled(self, region: Region) -> Region:
        """Scale a region from base resolution to frame resolution."""
        return region.scale(self.scale_x, self.scale_y)

    @property
    def pot_region(self) -> Region:
        """Region containing the pot amount display."""
        return self._pot_region

    @property
    def community_cards_region(self) -> Region:
        """Region containing the community cards."""
        return self._community_cards_region

    @property
    def community_card_slots(self) -> list[Region]:
        """List of 5 fixed regions for individual community cards."""
        return self._community_card_slots

    @property
    def hero_cards_region(self) -> Region:
        """Region containing the hero's hole cards."""
        return self._hero_cards_region

    @property
    def hero_card_slots(self) -> list[Region]:
        """List of 2 fixed regions for hero's hole cards."""
        return self._hero_card_slots

    @property
    def action_buttons_region(self) -> Region:
        """Region containing action buttons when visible."""
        return self._action_buttons_region

    @property
    def dealer_button_search_region(self) -> Region:
        """Region to search for the dealer button."""
        return self._dealer_button_search_region

    @property
    def balance_region(self) -> Region:
        """Region containing the player's total balance."""
        return self._balance_region

    def get_player_region(self, position: PlayerPosition) -> PlayerRegions:
        """Get the regions for a specific player position."""
        return self._player_regions[position]

    def get_all_player_regions(self) -> dict[PlayerPosition, PlayerRegions]:
        """Get all player regions."""
        return self._player_regions.copy()

    def extract_pot(self, frame: np.ndarray) -> np.ndarray:
        """Extract the pot display region from a frame."""
        return self._pot_region.extract(frame)

    def extract_community_cards(self, frame: np.ndarray) -> np.ndarray:
        """Extract the community cards region from a frame."""
        return self._community_cards_region.extract(frame)

    def extract_community_card_slots(self, frame: np.ndarray) -> list[np.ndarray]:
        """Extract the 5 individual community card slot images."""
        return [slot.extract(frame) for slot in self._community_card_slots]

    def extract_hero_cards(self, frame: np.ndarray) -> np.ndarray:
        """Extract the hero's hole cards region from a frame."""
        return self._hero_cards_region.extract(frame)

    def extract_hero_card_slots(self, frame: np.ndarray) -> list[np.ndarray]:
        """Extract the 2 individual hero card slot images."""
        return [slot.extract(frame) for slot in self._hero_card_slots]

    def extract_player_chips(self, frame: np.ndarray, position: PlayerPosition) -> np.ndarray:
        """Extract the chip count region for a player."""
        return self._player_regions[position].name_chip_box.extract(frame)

    def draw_regions(self, frame: np.ndarray, include_players: bool = True) -> np.ndarray:
        """
        Draw all regions on a frame for debugging/visualization.

        Args:
            frame: The frame to draw on.
            include_players: Whether to include player regions.

        Returns:
            Frame with regions drawn.
        """
        output = frame.copy()

        # Colors for different region types
        POT_COLOR = (0, 255, 0)       # Green
        CARDS_COLOR = (255, 0, 0)     # Blue
        PLAYER_COLOR = (0, 255, 255)  # Yellow
        ACTION_COLOR = (255, 0, 255)  # Magenta

        # Draw main regions
        self._draw_region(output, self._pot_region, POT_COLOR, "POT")
        self._draw_region(output, self._community_cards_region, CARDS_COLOR, "COMMUNITY")
        self._draw_region(output, self._hero_cards_region, CARDS_COLOR, "HERO CARDS")
        self._draw_region(output, self._action_buttons_region, ACTION_COLOR, "ACTIONS")

        # Draw player regions
        if include_players:
            for pos, player in self._player_regions.items():
                label = pos.name
                self._draw_region(output, player.name_chip_box, PLAYER_COLOR, label)

        return output

    def _draw_region(
        self,
        frame: np.ndarray,
        region: Region,
        color: Tuple[int, int, int],
        label: str
    ) -> None:
        """Draw a single region with label."""
        cv2.rectangle(frame, (region.x, region.y), (region.x2, region.y2), color, 2)
        cv2.putText(
            frame, label, (region.x, region.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )


def detect_table_regions(frame: np.ndarray) -> TableRegionDetector:
    """
    Create a TableRegionDetector for the given frame.

    Args:
        frame: A video frame from Governor of Poker.

    Returns:
        TableRegionDetector configured for the frame's dimensions.
    """
    height, width = frame.shape[:2]
    return TableRegionDetector(width, height)
