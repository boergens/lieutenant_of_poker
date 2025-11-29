#!/usr/bin/env python3
"""
Extract card slot images from a video frame where all 5 community cards are visible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from lieutenant_of_poker.frame_extractor import VideoFrameExtractor
from lieutenant_of_poker.table_regions import detect_table_regions
from lieutenant_of_poker.card_detector import CardDetector

# Load the table background color
TABLE_COLOR_IMAGE = Path.home() / "Desktop" / "color.png"


def load_table_color():
    """Load the average table background color."""
    if TABLE_COLOR_IMAGE.exists():
        img = cv2.imread(str(TABLE_COLOR_IMAGE))
        if img is not None:
            return np.mean(img, axis=(0, 1))
    return np.array([204, 96, 184])


def is_slot_filled(slot_image: np.ndarray, table_color: np.ndarray, threshold: int = 60) -> bool:
    """Check if a card slot has a card (not just table background)."""
    avg_color = np.mean(slot_image, axis=(0, 1))
    distance = np.linalg.norm(avg_color - table_color)
    return distance > threshold


def find_river_frame(video_path: str, start_time: float = 0, max_frames: int = 5000):
    """
    Find a frame where all 5 community card slots are filled.

    Returns (frame, timestamp_ms) or (None, None) if not found.
    """
    table_color = load_table_color()

    with VideoFrameExtractor(video_path) as video:
        duration_ms = video.duration_seconds * 1000
        print(f"Video: {video_path}")
        print(f"Resolution: {video.width}x{video.height}")
        print(f"Duration: {video.duration_seconds:.1f}s")
        print(f"Searching for river frame...")

        # Sample every 500ms
        sample_interval_ms = 500
        start_ms = start_time * 1000

        frames_checked = 0
        timestamp_ms = start_ms

        while timestamp_ms < duration_ms and frames_checked < max_frames:
            frame_info = video.get_frame_at_timestamp(timestamp_ms)
            if frame_info is None:
                timestamp_ms += sample_interval_ms
                continue

            frame = frame_info.image

            # Scale frame if needed
            h, w = frame.shape[:2]
            if w > 1920:
                scale = 1920 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Get card slots
            region_detector = detect_table_regions(frame)
            slots = region_detector.extract_community_card_slots(frame)

            # Check how many slots are filled
            filled_count = sum(1 for slot in slots if is_slot_filled(slot, table_color))

            if frames_checked % 20 == 0:
                print(f"  {timestamp_ms/1000:.1f}s: {filled_count}/5 cards")

            if filled_count == 5:
                print(f"\nFound river at {timestamp_ms/1000:.1f}s!")
                return frame, timestamp_ms, slots

            timestamp_ms += sample_interval_ms
            frames_checked += 1

        print("No river frame found")
        return None, None, None


def find_card_bounds(slot_image: np.ndarray, table_color: np.ndarray, threshold: int = 60):
    """Find the bounding box of the card within a slot image."""
    h, w = slot_image.shape[:2]

    # Calculate color distance for each pixel
    diff = slot_image.astype(np.float32) - table_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    # Pixels far from table color are part of the card
    card_mask = (distance > threshold).astype(np.uint8) * 255

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, card_mask

    # Find the largest contour (should be the card)
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    return (x, y, cw, ch), card_mask


def analyze_slots(slots, output_dir: Path):
    """Analyze the card positions in each slot."""
    table_color = load_table_color()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing {len(slots)} card slots:")
    print(f"Table color (BGR): {table_color}")
    print()

    all_bounds = []

    for i, slot in enumerate(slots):
        h, w = slot.shape[:2]
        bounds, mask = find_card_bounds(slot, table_color)

        # Save the slot image
        cv2.imwrite(str(output_dir / f"slot_{i}_original.png"), slot)
        cv2.imwrite(str(output_dir / f"slot_{i}_mask.png"), mask)

        if bounds:
            x, y, cw, ch = bounds
            print(f"Slot {i}: slot_size={w}x{h}, card at ({x},{y}) size {cw}x{ch}")
            all_bounds.append(bounds)

            # Save with bounding box
            debug = slot.copy()
            cv2.rectangle(debug, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / f"slot_{i}_debug.png"), debug)

            # Save just the card
            card = slot[y:y+ch, x:x+cw]
            cv2.imwrite(str(output_dir / f"slot_{i}_card.png"), card)
        else:
            print(f"Slot {i}: slot_size={w}x{h}, NO CARD FOUND")

    if all_bounds:
        print()
        print("=== SUMMARY ===")
        xs = [b[0] for b in all_bounds]
        ys = [b[1] for b in all_bounds]
        ws = [b[2] for b in all_bounds]
        hs = [b[3] for b in all_bounds]

        print(f"X positions: {xs} (std: {np.std(xs):.1f})")
        print(f"Y positions: {ys} (std: {np.std(ys):.1f})")
        print(f"Widths: {ws} (std: {np.std(ws):.1f})")
        print(f"Heights: {hs} (std: {np.std(hs):.1f})")
        print()
        print(f"Recommended uniform crop:")
        print(f"  x = {int(np.mean(xs))}")
        print(f"  y = {int(np.mean(ys))}")
        print(f"  width = {int(np.mean(ws))}")
        print(f"  height = {int(np.mean(hs))}")

    return all_bounds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract card slots from video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", "-s", type=float, default=0,
                       help="Start time in seconds")
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/card_slots"),
                       help="Output directory")

    args = parser.parse_args()

    frame, timestamp_ms, slots = find_river_frame(args.video, args.start)

    if frame is not None:
        # Save the full frame
        args.output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / "full_frame.png"), frame)
        print(f"Saved full frame to {args.output / 'full_frame.png'}")

        # Analyze the slots
        analyze_slots(slots, args.output)
