"""
Dealer button detection for Governor of Poker.

Detects the circular "D" dealer button using template matching and determines
which player is currently the dealer based on button proximity to player positions.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .table_regions import Region, TableRegionDetector, seat_name


# Path to button template
BUTTON_TEMPLATE_PATH = Path(__file__).parent / "button_library" / "dealer_button.png"

# Search region covering the table area where button can appear (base resolution 1728x1117)
SEARCH_REGION_BASE = Region(x=250, y=250, width=1250, height=550)

# Template matching threshold (lower = stricter match)
MATCH_THRESHOLD = 0.70


class DealerDetector:
    """Detects dealer button using template matching."""

    def __init__(self):
        """Initialize the detector and load the button template."""
        self.template = self._load_template()
        if self.template is not None:
            self.template_h, self.template_w = self.template.shape[:2]
        else:
            self.template_h, self.template_w = 0, 0

    def _load_template(self) -> Optional[np.ndarray]:
        """Load the dealer button template image."""
        if BUTTON_TEMPLATE_PATH.exists():
            template = cv2.imread(str(BUTTON_TEMPLATE_PATH))
            return template
        return None

    def detect(
        self,
        frame: np.ndarray,
        region_detector: Optional[TableRegionDetector] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Detect the dealer button position in a frame.

        Args:
            frame: BGR image of the poker table.
            region_detector: Optional TableRegionDetector for scaling.

        Returns:
            (x, y) center coordinates of the button, or None if not found.
        """
        if self.template is None:
            return None

        h, w = frame.shape[:2]

        # Scale search region to frame size
        scale_x = w / 1728
        scale_y = h / 1117
        search_region = SEARCH_REGION_BASE.scale(scale_x, scale_y)

        # Extract search region
        roi = search_region.extract(frame)

        # Scale template if needed
        template = self.template
        if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
            new_w = int(self.template_w * scale_x)
            new_h = int(self.template_h * scale_y)
            if new_w > 0 and new_h > 0:
                template = cv2.resize(self.template, (new_w, new_h))

        # Template matching
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= MATCH_THRESHOLD:
            # Get center of matched region
            th, tw = template.shape[:2]
            cx = search_region.x + max_loc[0] + tw // 2
            cy = search_region.y + max_loc[1] + th // 2
            return (cx, cy)

        return None


# Global detector instance
_detector: Optional[DealerDetector] = None


def _get_detector() -> DealerDetector:
    """Get or create the global detector instance."""
    global _detector
    if _detector is None:
        _detector = DealerDetector()
    return _detector


def detect_dealer_button(
    frame: np.ndarray,
    region_detector: Optional[TableRegionDetector] = None
) -> Optional[Tuple[int, int]]:
    """
    Detect the dealer button position in a frame.

    Args:
        frame: BGR image of the poker table.
        region_detector: Optional TableRegionDetector for scaling.

    Returns:
        (x, y) center coordinates of the button, or None if not found.
    """
    return _get_detector().detect(frame, region_detector)


def detect_dealer_position(
    frame: np.ndarray,
    region_detector: Optional[TableRegionDetector] = None
) -> Optional[int]:
    """
    Detect which player position has the dealer button.

    Args:
        frame: BGR image of the poker table.
        region_detector: Optional TableRegionDetector (will create if not provided).

    Returns:
        Seat index (0-4) of the dealer, or None if button not found.
    """
    h, w = frame.shape[:2]

    if region_detector is None:
        region_detector = TableRegionDetector(w, h)

    # Find the button
    button_pos = detect_dealer_button(frame, region_detector)
    if button_pos is None:
        return None

    button_x, button_y = button_pos

    # Get player regions and find closest
    player_regions = region_detector.get_all_player_regions()

    min_dist = float('inf')
    closest_position = None

    for position, regions in player_regions.items():
        # Use the center of the name_chip_box as the player anchor
        player_center = regions.name_chip_box.center

        # Calculate distance
        dist = np.sqrt(
            (button_x - player_center[0]) ** 2 +
            (button_y - player_center[1]) ** 2
        )

        if dist < min_dist:
            min_dist = dist
            closest_position = position

    return closest_position


def draw_dealer_detection(
    frame: np.ndarray,
    button_pos: Optional[Tuple[int, int]] = None,
    dealer_position: Optional[int] = None
) -> np.ndarray:
    """
    Draw dealer button detection visualization on a frame.

    Args:
        frame: BGR image to draw on.
        button_pos: Optional pre-computed button position.
        dealer_position: Optional pre-computed dealer seat index (0-4).

    Returns:
        Frame with visualization drawn.
    """
    output = frame.copy()

    # Detect if not provided
    if button_pos is None:
        button_pos = detect_dealer_button(frame)

    if button_pos is not None:
        # Draw button location
        cv2.circle(output, button_pos, 25, (0, 255, 0), 2)
        cv2.circle(output, button_pos, 3, (0, 255, 0), -1)

        # Draw label
        if dealer_position is None:
            dealer_position = detect_dealer_position(frame)

        if dealer_position is not None:
            label = f"Dealer: {seat_name(dealer_position)}"
            cv2.putText(
                output, label, (button_pos[0] + 30, button_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    return output
