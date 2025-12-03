#!/usr/bin/env python3
"""Remove all files from unmatched folders in the card library."""

import shutil
from pathlib import Path


def clear_unmatched_folders():
    """Find and clear all unmatched folders in card_library."""
    card_library = Path(__file__).parent.parent / "lieutenant_of_poker" / "card_library"

    if not card_library.exists():
        print(f"Card library not found: {card_library}")
        return

    unmatched_dirs = list(card_library.rglob("unmatched"))

    if not unmatched_dirs:
        print("No unmatched folders found.")
        return

    total_removed = 0
    for unmatched_dir in unmatched_dirs:
        if unmatched_dir.is_dir():
            files = list(unmatched_dir.iterdir())
            count = len(files)
            if count > 0:
                for f in files:
                    f.unlink()
                print(f"Removed {count} files from {unmatched_dir.relative_to(card_library)}")
                total_removed += count

    print(f"\nTotal: {total_removed} files removed")


if __name__ == "__main__":
    clear_unmatched_folders()
