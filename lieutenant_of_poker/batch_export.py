"""Batch export hand histories from a folder of videos."""

import sys
from pathlib import Path
from typing import Optional

from .export import export_video


VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


def batch_export(
    folder: str,
    output_dir: Optional[str],
    fmt: str,
    extension: str,
    max_rake_pct: float = 0.10,
):
    """Export all videos in folder to text files.

    Args:
        folder: Path to folder containing video files
        output_dir: Path to output directory for text files (None = same as folder)
        fmt: Export format (snowie, pokerstars, human, actions)
        extension: Output file extension
        max_rake_pct: Maximum rake as percentage of pot (default 10%, 0 to disable)
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        return

    out_path = Path(output_dir) if output_dir else folder_path
    out_path.mkdir(parents=True, exist_ok=True)

    videos = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print(f"No video files found in {folder_path}", file=sys.stderr)
        return

    print(f"Found {len(videos)} video(s) in {folder_path}", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)
    print(f"Format: {fmt}", file=sys.stderr)
    print(file=sys.stderr)

    success = errors = 0

    for i, video in enumerate(videos, 1):
        out_file = out_path / (video.stem + extension)
        print(f"[{i}/{len(videos)}] {video.name}", file=sys.stderr, end=" ")

        try:
            output = export_video(str(video), fmt, max_rake_pct)

            if not output:
                print("-> no hands", file=sys.stderr)
                continue

            out_file.write_text(output)
            print(f"-> {out_file.name}", file=sys.stderr)
            success += 1

        except Exception as e:
            print(f"-> ERROR: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone! {success} exported, {errors} errors", file=sys.stderr)
