#!/usr/bin/env python3
"""List all available windows that can be captured."""

from lieutenant_of_poker.screen_capture import ScreenCaptureKitCapture

def main():
    capture = ScreenCaptureKitCapture()
    windows = capture.list_windows()

    print(f"Found {len(windows)} windows:\n")
    for w in windows:
        print(f"  {w.owner_name}: {w.title}")
        print(f"    ID: {w.window_id}, Size: {w.bounds[2]}x{w.bounds[3]}")
        print()

if __name__ == "__main__":
    main()
