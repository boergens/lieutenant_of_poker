#!/usr/bin/env python3
"""
GUI calibration tool for marking table regions on a frame.

Usage:
    python -m lieutenant_of_poker.calibrate video.mp4 --timestamp 60

Controls:
    - Click and drag to draw a region
    - Number keys 1-9 to select player position
    - 'p' for pot, 'c' for community cards, 'h' for hero cards
    - 's' to save and print coordinates
    - 'r' to reset current region
    - 'q' to quit
    - Arrow keys to fine-tune selected region
"""

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class RegionType(Enum):
    POT = "pot"
    COMMUNITY = "community"
    HERO = "hero"
    PLAYER_1 = "player_1"  # TOP_LEFT
    PLAYER_2 = "player_2"  # TOP
    PLAYER_3 = "player_3"  # TOP_RIGHT
    PLAYER_4 = "player_4"  # RIGHT
    PLAYER_5 = "player_5"  # BOTTOM_RIGHT
    PLAYER_6 = "player_6"  # BOTTOM (hero)
    PLAYER_7 = "player_7"  # BOTTOM_LEFT
    PLAYER_8 = "player_8"  # LEFT
    PLAYER_9 = "player_9"  # CENTER


@dataclass
class Region:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self):
        return (self.x, self.y, self.width, self.height)


class CalibrationTool:
    """Interactive GUI for calibrating table regions."""

    # Colors for different region types
    COLORS = {
        RegionType.POT: (0, 255, 0),         # Green
        RegionType.COMMUNITY: (255, 0, 0),    # Blue
        RegionType.HERO: (0, 255, 255),       # Yellow
        RegionType.PLAYER_1: (255, 128, 0),   # Orange
        RegionType.PLAYER_2: (255, 128, 0),
        RegionType.PLAYER_3: (255, 128, 0),
        RegionType.PLAYER_4: (255, 128, 0),
        RegionType.PLAYER_5: (255, 128, 0),
        RegionType.PLAYER_6: (255, 128, 0),
        RegionType.PLAYER_7: (255, 128, 0),
        RegionType.PLAYER_8: (255, 128, 0),
        RegionType.PLAYER_9: (255, 128, 0),
    }

    POSITION_NAMES = {
        RegionType.PLAYER_1: "TOP_LEFT",
        RegionType.PLAYER_2: "TOP",
        RegionType.PLAYER_3: "TOP_RIGHT",
        RegionType.PLAYER_4: "RIGHT",
        RegionType.PLAYER_5: "BOTTOM_RIGHT",
        RegionType.PLAYER_6: "BOTTOM (hero)",
        RegionType.PLAYER_7: "BOTTOM_LEFT",
        RegionType.PLAYER_8: "LEFT",
        RegionType.PLAYER_9: "CENTER",
    }

    def __init__(self, frame: np.ndarray, scale: float = 1.0):
        """
        Initialize the calibration tool.

        Args:
            frame: The video frame to calibrate on.
            scale: Display scale factor (for large frames).
        """
        self.original_frame = frame.copy()
        self.scale = scale

        # Scale frame for display if needed
        if scale != 1.0:
            h, w = frame.shape[:2]
            self.display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            self.display_frame = frame.copy()

        self.working_frame = self.display_frame.copy()

        # Region storage
        self.regions: dict[RegionType, Region] = {}

        # Current selection state
        self.current_type: RegionType = RegionType.POT
        self.drawing = False
        self.start_point: Optional[tuple[int, int]] = None
        self.end_point: Optional[tuple[int, int]] = None

        # Window name
        self.window_name = "Region Calibration - Press 'h' for help"

    def _to_original_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert display coordinates to original frame coordinates."""
        return int(x / self.scale), int(y / self.scale)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                # Convert to original coordinates
                x1, y1 = self._to_original_coords(*self.start_point)
                x2, y2 = self._to_original_coords(*self.end_point)

                # Normalize coordinates
                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 5 and height > 5:  # Minimum size
                    self.regions[self.current_type] = Region(left, top, width, height)
                    print(f"Set {self.current_type.value}: x={left}, y={top}, w={width}, h={height}")

                self.start_point = None
                self.end_point = None
                self._redraw()

    def _redraw(self):
        """Redraw the frame with all regions."""
        self.working_frame = self.display_frame.copy()

        # Draw all saved regions
        for region_type, region in self.regions.items():
            color = self.COLORS.get(region_type, (255, 255, 255))

            # Scale coordinates for display
            x = int(region.x * self.scale)
            y = int(region.y * self.scale)
            w = int(region.width * self.scale)
            h = int(region.height * self.scale)

            # Draw rectangle
            thickness = 3 if region_type == self.current_type else 2
            cv2.rectangle(self.working_frame, (x, y), (x + w, y + h), color, thickness)

            # Draw label
            label = region_type.value
            if region_type in self.POSITION_NAMES:
                label = self.POSITION_NAMES[region_type]
            cv2.putText(self.working_frame, label, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw current selection in progress
        if self.drawing and self.start_point and self.end_point:
            color = self.COLORS.get(self.current_type, (255, 255, 255))
            cv2.rectangle(self.working_frame, self.start_point, self.end_point, color, 2)

        # Draw current mode indicator
        mode_text = f"Mode: {self.current_type.value}"
        if self.current_type in self.POSITION_NAMES:
            mode_text += f" ({self.POSITION_NAMES[self.current_type]})"
        cv2.putText(self.working_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(self.window_name, self.working_frame)

    def _print_help(self):
        """Print help text."""
        print("\n=== Calibration Tool Help ===")
        print("Mouse:")
        print("  Click and drag to draw a region")
        print("\nKeys:")
        print("  p - Select POT region")
        print("  c - Select COMMUNITY CARDS region")
        print("  h - Select HERO CARDS region")
        print("  1-9 - Select player position (1=TOP_LEFT, 2=TOP, ...)")
        print("  Arrow keys - Fine-tune selected region position")
        print("  +/- - Fine-tune region size")
        print("  r - Reset current region")
        print("  s - Save and print all coordinates")
        print("  q - Quit")
        print("==============================\n")

    def _adjust_region(self, dx: int, dy: int, dw: int = 0, dh: int = 0):
        """Adjust the current region."""
        if self.current_type in self.regions:
            r = self.regions[self.current_type]
            self.regions[self.current_type] = Region(
                max(0, r.x + dx),
                max(0, r.y + dy),
                max(10, r.width + dw),
                max(10, r.height + dh)
            )
            self._redraw()

    def _print_regions(self):
        """Print all regions in a format suitable for code."""
        print("\n" + "=" * 60)
        print("# Region coordinates (at original resolution)")
        print("=" * 60)

        # Base regions
        if RegionType.POT in self.regions:
            r = self.regions[RegionType.POT]
            print(f"\n# Pot region")
            print(f"self._pot_region = self._scaled(Region(x={r.x}, y={r.y}, width={r.width}, height={r.height}))")

        if RegionType.COMMUNITY in self.regions:
            r = self.regions[RegionType.COMMUNITY]
            print(f"\n# Community cards region")
            print(f"self._community_cards_region = self._scaled(Region(x={r.x}, y={r.y}, width={r.width}, height={r.height}))")

        if RegionType.HERO in self.regions:
            r = self.regions[RegionType.HERO]
            print(f"\n# Hero cards region")
            print(f"self._hero_cards_region = self._scaled(Region(x={r.x}, y={r.y}, width={r.width}, height={r.height}))")

        # Player regions
        position_mapping = {
            RegionType.PLAYER_1: "PlayerPosition.TOP_LEFT",
            RegionType.PLAYER_2: "PlayerPosition.TOP",
            RegionType.PLAYER_3: "PlayerPosition.TOP_RIGHT",
            RegionType.PLAYER_4: "PlayerPosition.RIGHT",
            RegionType.PLAYER_5: "PlayerPosition.BOTTOM_RIGHT",
            RegionType.PLAYER_6: "PlayerPosition.BOTTOM",
            RegionType.PLAYER_7: "PlayerPosition.BOTTOM_LEFT",
            RegionType.PLAYER_8: "PlayerPosition.LEFT",
            RegionType.PLAYER_9: "PlayerPosition.CENTER",
        }

        print("\n# Player regions")
        for region_type, position_name in position_mapping.items():
            if region_type in self.regions:
                r = self.regions[region_type]
                print(f"# {position_name}")
                print(f"# name_chip_box: Region(x={r.x}, y={r.y}, width={r.width}, height={r.height})")

        print("\n" + "=" * 60)

    def run(self):
        """Run the calibration tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._print_help()
        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('h'):
                self._print_help()
            elif key == ord('p'):
                self.current_type = RegionType.POT
                print(f"Selected: POT")
                self._redraw()
            elif key == ord('c'):
                self.current_type = RegionType.COMMUNITY
                print(f"Selected: COMMUNITY CARDS")
                self._redraw()
            elif key == ord('o'):  # 'o' for hOle cards since 'h' is help
                self.current_type = RegionType.HERO
                print(f"Selected: HERO CARDS")
                self._redraw()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'),
                        ord('6'), ord('7'), ord('8'), ord('9')]:
                player_num = key - ord('0')
                self.current_type = RegionType(f"player_{player_num}")
                pos_name = self.POSITION_NAMES.get(self.current_type, "")
                print(f"Selected: PLAYER {player_num} ({pos_name})")
                self._redraw()
            elif key == ord('r'):
                if self.current_type in self.regions:
                    del self.regions[self.current_type]
                    print(f"Reset {self.current_type.value}")
                    self._redraw()
            elif key == ord('s'):
                self._print_regions()
            elif key == 81 or key == 2:  # Left arrow
                self._adjust_region(-5, 0)
            elif key == 83 or key == 3:  # Right arrow
                self._adjust_region(5, 0)
            elif key == 82 or key == 0:  # Up arrow
                self._adjust_region(0, -5)
            elif key == 84 or key == 1:  # Down arrow
                self._adjust_region(0, 5)
            elif key == ord('+') or key == ord('='):
                self._adjust_region(0, 0, 5, 5)
            elif key == ord('-'):
                self._adjust_region(0, 0, -5, -5)

        cv2.destroyAllWindows()
        return self.regions


def main():
    parser = argparse.ArgumentParser(
        description="GUI tool for calibrating table regions"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--frame", "-f", type=int, default=None,
        help="Frame number to use"
    )
    parser.add_argument(
        "--timestamp", "-t", type=float, default=None,
        help="Timestamp in seconds"
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=None,
        help="Display scale factor (auto-calculated if not specified)"
    )

    args = parser.parse_args()

    # Load frame
    from lieutenant_of_poker.frame_extractor import VideoFrameExtractor

    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {args.video}")
        print(f"Resolution: {video.width}x{video.height}")

        if args.frame is not None:
            frame_info = video.get_frame_at(args.frame)
        elif args.timestamp is not None:
            frame_info = video.get_frame_at_timestamp(args.timestamp * 1000)
        else:
            frame_info = video.get_frame_at(0)

        if frame_info is None:
            print("Error: Could not read frame", file=sys.stderr)
            sys.exit(1)

        frame = frame_info.image

        # Auto-calculate scale to fit on screen
        if args.scale is None:
            # Target max dimension of 1400 pixels for display
            max_dim = max(video.width, video.height)
            if max_dim > 1400:
                scale = 1400 / max_dim
            else:
                scale = 1.0
        else:
            scale = args.scale

        print(f"Display scale: {scale:.2f}")
        print(f"\nStarting calibration tool...")
        print("Press 'h' in the window for help\n")

        tool = CalibrationTool(frame, scale=scale)
        regions = tool.run()

        if regions:
            print("\nFinal regions captured:")
            tool._print_regions()


if __name__ == "__main__":
    main()
