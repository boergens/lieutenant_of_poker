#!/usr/bin/env python3
"""
GUI tool for marking rank and suit regions within the hero cards region.

Usage:
    python -m lieutenant_of_poker.calibrate_hero video.mp4 --timestamp 60

Controls:
    - Click and drag to draw a region
    - 1 = Left card RANK
    - 2 = Left card SUIT
    - 3 = Right card RANK
    - 4 = Right card SUIT
    - 'p' to print coordinates
    - 'q' to quit
"""

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SubregionType(Enum):
    LEFT_RANK = "left_rank"
    LEFT_SUIT = "left_suit"
    RIGHT_RANK = "right_rank"
    RIGHT_SUIT = "right_suit"


@dataclass
class Region:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self):
        return (self.x, self.y, self.width, self.height)


class HeroCardCalibrationTool:
    """Interactive GUI for calibrating rank/suit regions within hero cards region."""

    COLORS = {
        SubregionType.LEFT_RANK: (0, 255, 0),     # Green
        SubregionType.LEFT_SUIT: (0, 255, 255),   # Yellow
        SubregionType.RIGHT_RANK: (255, 0, 0),    # Blue
        SubregionType.RIGHT_SUIT: (255, 0, 255),  # Magenta
    }

    LABELS = {
        SubregionType.LEFT_RANK: "1: L-RANK",
        SubregionType.LEFT_SUIT: "2: L-SUIT",
        SubregionType.RIGHT_RANK: "3: R-RANK",
        SubregionType.RIGHT_SUIT: "4: R-SUIT",
    }

    def __init__(self, hero_region: np.ndarray, scale: float = 3.0):
        """
        Initialize the tool with the full hero cards region.

        Args:
            hero_region: Image of the full hero cards region (both cards).
            scale: Display scale factor.
        """
        self.original = hero_region.copy()
        self.scale = scale

        # Scale for display
        h, w = hero_region.shape[:2]
        self.display = cv2.resize(
            hero_region, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_NEAREST
        )

        # Region storage
        self.regions: dict[SubregionType, Region] = {}

        # Current selection state
        self.current_type: SubregionType = SubregionType.LEFT_RANK
        self.drawing = False
        self.start_point: Optional[tuple[int, int]] = None
        self.end_point: Optional[tuple[int, int]] = None

        # Window name
        self.window_name = "Hero Card Calibration"

    def _to_original_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert display coordinates to original coordinates."""
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

                if width > 2 and height > 2:  # Minimum size
                    self.regions[self.current_type] = Region(left, top, width, height)
                    print(f"Set {self.current_type.value}: x={left}, y={top}, w={width}, h={height}")

                self.start_point = None
                self.end_point = None
                self._redraw()

    def _redraw(self):
        """Redraw the image with regions."""
        display = self.display.copy()

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
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)

            # Draw label
            label = self.LABELS.get(region_type, region_type.value)
            cv2.putText(display, label, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw current selection in progress
        if self.drawing and self.start_point and self.end_point:
            color = self.COLORS.get(self.current_type, (255, 255, 255))
            cv2.rectangle(display, self.start_point, self.end_point, color, 2)

        # Draw status bar at bottom
        disp_h, disp_w = display.shape[:2]
        status_h = 50
        status_bar = np.zeros((status_h, disp_w, 3), dtype=np.uint8)
        status_bar[:] = (50, 50, 50)

        current_label = self.LABELS.get(self.current_type, self.current_type.value)
        status_text = f"Current: {current_label} | Keys: 1=L-rank 2=L-suit 3=R-rank 4=R-suit | p=print q=quit"
        cv2.putText(status_bar, status_text, (10, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine
        combined = np.vstack([display, status_bar])
        cv2.imshow(self.window_name, combined)

    def _print_regions(self):
        """Print all regions in code format."""
        print("\n" + "=" * 60)
        print("# Hero card subregions (relative to hero_cards_region)")
        print("=" * 60)

        for region_type in SubregionType:
            if region_type in self.regions:
                r = self.regions[region_type]
                var_name = f"HERO_{region_type.value.upper()}_REGION"
                print(f"{var_name} = ({r.x}, {r.y}, {r.width}, {r.height})  # x, y, w, h")
            else:
                print(f"# {region_type.value} not defined")

        print("\n" + "=" * 60)

    def run(self):
        """Run the calibration tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n=== Hero Card Subregion Calibration ===")
        print("Mark all 4 regions within the hero cards area:")
        print("  1 = Left card RANK")
        print("  2 = Left card SUIT")
        print("  3 = Right card RANK")
        print("  4 = Right card SUIT")
        print("  p = Print coordinates")
        print("  q = Quit")
        print("=" * 40 + "\n")

        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_type = SubregionType.LEFT_RANK
                print(f"Selected: Left card RANK")
                self._redraw()
            elif key == ord('2'):
                self.current_type = SubregionType.LEFT_SUIT
                print(f"Selected: Left card SUIT")
                self._redraw()
            elif key == ord('3'):
                self.current_type = SubregionType.RIGHT_RANK
                print(f"Selected: Right card RANK")
                self._redraw()
            elif key == ord('4'):
                self.current_type = SubregionType.RIGHT_SUIT
                print(f"Selected: Right card SUIT")
                self._redraw()
            elif key == ord('p'):
                self._print_regions()

        cv2.destroyAllWindows()
        return self.regions


def main():
    parser = argparse.ArgumentParser(
        description="GUI tool for calibrating hero card rank/suit subregions"
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
        "--scale", "-s", type=float, default=3.0,
        help="Display scale factor (default: 3.0)"
    )

    args = parser.parse_args()

    # Load frame
    from lieutenant_of_poker.frame_extractor import VideoFrameExtractor
    from lieutenant_of_poker.table_regions import detect_table_regions

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

        # Extract the full hero cards region (not individual slots)
        region_detector = detect_table_regions(frame)
        hero_region = region_detector.extract_hero_cards(frame)

        print(f"Hero region size: {hero_region.shape[1]}x{hero_region.shape[0]}")
        print(f"\nStarting hero card calibration tool...")

        tool = HeroCardCalibrationTool(hero_region, scale=args.scale)
        regions = tool.run()

        if regions:
            print("\nFinal regions captured:")
            tool._print_regions()


if __name__ == "__main__":
    main()
