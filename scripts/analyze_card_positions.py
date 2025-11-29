#!/usr/bin/env python3
"""
Analyze card positions in slot images to determine precise cropping.

Uses the table background color to threshold and find exact card boundaries.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load the table background color
TABLE_COLOR_IMAGE = Path.home() / "Desktop" / "color.png"


def load_table_color():
    """Load the average table background color."""
    if TABLE_COLOR_IMAGE.exists():
        img = cv2.imread(str(TABLE_COLOR_IMAGE))
        if img is not None:
            return np.mean(img, axis=(0, 1))
    return np.array([204, 96, 184])  # Default purple


def find_card_bounds(slot_image: np.ndarray, table_color: np.ndarray, threshold: int = 60):
    """
    Find the bounding box of the card within a slot image.

    Args:
        slot_image: BGR image of the card slot
        table_color: BGR color of the table background
        threshold: Color distance threshold for background detection

    Returns:
        (x, y, w, h) bounding box of the card, or None if not found
    """
    h, w = slot_image.shape[:2]

    # Create a mask where pixels are NOT the table background
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


def analyze_slot_images(library_dir: Path, output_dir: Path = None):
    """
    Analyze all card images in the library to find consistent card positions.
    """
    table_color = load_table_color()
    print(f"Table background color (BGR): {table_color}")
    print()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Group images by slot
    slot_images = {}
    for image_path in library_dir.glob("*.png"):
        name = image_path.stem
        parts = name.split("_")
        if len(parts) >= 3 and parts[2].startswith("slot"):
            try:
                slot_idx = int(parts[2][4:])
                if slot_idx not in slot_images:
                    slot_images[slot_idx] = []
                slot_images[slot_idx].append(image_path)
            except ValueError:
                pass

    print(f"Found {sum(len(v) for v in slot_images.values())} images across {len(slot_images)} slots")
    print()

    # Analyze each slot
    all_bounds = {}
    for slot_idx in sorted(slot_images.keys()):
        print(f"=== Slot {slot_idx} ===")
        slot_bounds = []

        for image_path in slot_images[slot_idx]:
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            bounds, mask = find_card_bounds(img, table_color)

            if bounds:
                x, y, cw, ch = bounds
                print(f"  {image_path.name}: size={w}x{h}, card at ({x},{y}) size {cw}x{ch}")
                slot_bounds.append(bounds)

                # Save debug images
                if output_dir:
                    # Draw bounding box
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                    cv2.imwrite(str(output_dir / f"debug_{image_path.name}"), debug_img)
                    cv2.imwrite(str(output_dir / f"mask_{image_path.name}"), mask)
            else:
                print(f"  {image_path.name}: size={w}x{h}, NO CARD FOUND")

        if slot_bounds:
            # Calculate average bounds for this slot
            avg_x = int(np.mean([b[0] for b in slot_bounds]))
            avg_y = int(np.mean([b[1] for b in slot_bounds]))
            avg_w = int(np.mean([b[2] for b in slot_bounds]))
            avg_h = int(np.mean([b[3] for b in slot_bounds]))
            print(f"  Average: ({avg_x},{avg_y}) size {avg_w}x{avg_h}")
            all_bounds[slot_idx] = (avg_x, avg_y, avg_w, avg_h)
        print()

    # Summary
    print("=== SUMMARY ===")
    print("Recommended card crop regions per slot:")
    for slot_idx, (x, y, w, h) in sorted(all_bounds.items()):
        print(f"  Slot {slot_idx}: x={x}, y={y}, width={w}, height={h}")

    # Check consistency
    if all_bounds:
        widths = [b[2] for b in all_bounds.values()]
        heights = [b[3] for b in all_bounds.values()]
        print()
        print(f"Width range: {min(widths)} - {max(widths)} (variance: {np.std(widths):.1f})")
        print(f"Height range: {min(heights)} - {max(heights)} (variance: {np.std(heights):.1f})")

    return all_bounds


def extract_cards_precisely(library_dir: Path, output_dir: Path, bounds_per_slot: dict):
    """
    Re-extract cards using precise bounds, saving normalized versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the most common card size
    all_widths = [b[2] for b in bounds_per_slot.values()]
    all_heights = [b[3] for b in bounds_per_slot.values()]
    target_w = int(np.median(all_widths))
    target_h = int(np.median(all_heights))
    print(f"Target card size: {target_w}x{target_h}")

    for image_path in library_dir.glob("*.png"):
        name = image_path.stem
        parts = name.split("_")
        if len(parts) < 3 or not parts[2].startswith("slot"):
            continue

        try:
            slot_idx = int(parts[2][4:])
        except ValueError:
            continue

        if slot_idx not in bounds_per_slot:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        x, y, w, h = bounds_per_slot[slot_idx]

        # Extract and resize to consistent size
        card = img[y:y+h, x:x+w]
        card_resized = cv2.resize(card, (target_w, target_h), interpolation=cv2.INTER_AREA)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), card_resized)
        print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze card positions in slot images")
    parser.add_argument("--library", "-l", type=Path,
                       default=Path(__file__).parent.parent / "lieutenant_of_poker" / "card_library",
                       help="Path to card library directory")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output directory for debug images")
    parser.add_argument("--extract", "-e", type=Path,
                       help="Extract precisely cropped cards to this directory")

    args = parser.parse_args()

    print(f"Analyzing card library: {args.library}")
    print()

    bounds = analyze_slot_images(args.library, args.output)

    if args.extract and bounds:
        print()
        print(f"Extracting precisely cropped cards to: {args.extract}")
        extract_cards_precisely(args.library, args.extract, bounds)
