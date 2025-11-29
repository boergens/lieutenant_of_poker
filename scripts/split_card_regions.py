#!/usr/bin/env python3
"""
Test script to find optimal rank and suit crop regions within a card slot.
"""

import cv2
import numpy as np
from pathlib import Path

# Load a sample card slot
slot_dir = Path("/tmp/card_slots")

for i in range(5):
    slot_path = slot_dir / f"slot_{i}_original.png"
    if not slot_path.exists():
        continue

    img = cv2.imread(str(slot_path))
    h, w = img.shape[:2]
    print(f"Slot {i}: {w}x{h}")

    # Based on analysis: card is at roughly (3, 10) with size 96x127
    # Rank is in top-left corner of the card
    # Suit symbol is below the rank

    # Rank region: top-left of the card area
    # Shifted more right/down, made larger
    rank_x = 10
    rank_y = 15
    rank_w = 55
    rank_h = 55

    rank_region = img[rank_y:rank_y+rank_h, rank_x:rank_x+rank_w]

    # Suit region: below the rank
    # Shifted more right/down
    suit_x = 30
    suit_y = 75
    suit_w = 60
    suit_h = 55

    suit_region = img[suit_y:suit_y+suit_h, suit_x:suit_x+suit_w]

    # Save the regions
    output_dir = Path("/tmp/card_regions")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / f"slot_{i}_rank.png"), rank_region)
    cv2.imwrite(str(output_dir / f"slot_{i}_suit.png"), suit_region)

    # Also save with rectangles drawn
    debug = img.copy()
    cv2.rectangle(debug, (rank_x, rank_y), (rank_x+rank_w, rank_y+rank_h), (0, 255, 0), 2)
    cv2.rectangle(debug, (suit_x, suit_y), (suit_x+suit_w, suit_y+suit_h), (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / f"slot_{i}_debug.png"), debug)

print(f"\nSaved to {output_dir}")
print("Green = rank region, Red = suit region")
