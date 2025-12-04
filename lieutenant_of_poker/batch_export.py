"""Batch export hand histories from a folder of videos."""

import re
import sys
from pathlib import Path

from .analysis import analyze_video, AnalysisConfig
from .frame_extractor import get_video_info
from .first_frame import detect_from_video
from .snowie_export import export_snowie
from .pokerstars_export import export_pokerstars
from .human_export import export_human
from .action_log_export import export_action_log


VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}

# Matches gop3_YYYYMMDD_HHMMSS format
TIMESTAMP_PATTERN = re.compile(r'_(\d{8})_(\d{6})')


def extract_hand_id(filename: str) -> str | None:
    """Extract hand ID from filename like gop3_20251203_095609.mp4 -> 20251203095609."""
    match = TIMESTAMP_PATTERN.search(filename)
    if match:
        return match.group(1) + match.group(2)
    return None


def batch_export(
    folder: Path,
    output_dir: Path,
    fmt: str,
    extension: str,
    table_background: str | None = None,
):
    """Export all videos in folder to text files.

    Args:
        folder: Path to folder containing video files
        output_dir: Path to output directory for text files
        fmt: Export format (snowie, pokerstars, human, actions)
        extension: Output file extension
        table_background: Optional path to table background image
    """
    videos = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print(f"No video files found in {folder}", file=sys.stderr)
        return

    print(f"Found {len(videos)} video(s) in {folder}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)
    print(f"Format: {fmt}", file=sys.stderr)
    if table_background:
        print(f"Table background: {table_background}", file=sys.stderr)
    print(file=sys.stderr)

    success = errors = 0
    config = AnalysisConfig(table_background=table_background)

    for i, video in enumerate(videos, 1):
        out_file = output_dir / (video.stem + extension)
        print(f"[{i}/{len(videos)}] {video.name}", file=sys.stderr, end=" ")

        try:
            first = detect_from_video(str(video), 0)
            button = first.button_index or 0
            names = first.player_names

            states = analyze_video(str(video), config)
            if not states:
                print("-> no hands", file=sys.stderr)
                continue

            if fmt == "snowie":
                hand_id = extract_hand_id(video.name)
                output = export_snowie(states, button, names, hand_id=hand_id)
            elif fmt == "human":
                output = export_human(states, button, names)
            elif fmt == "actions":
                output = export_action_log(states, button, names)
            else:
                output = export_pokerstars(states, button, names)

            if not output:
                print("-> empty", file=sys.stderr)
                continue

            out_file.write_text(output)
            print(f"-> {out_file.name}", file=sys.stderr)
            success += 1

        except Exception as e:
            print(f"-> ERROR: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone! {success} exported, {errors} errors", file=sys.stderr)
