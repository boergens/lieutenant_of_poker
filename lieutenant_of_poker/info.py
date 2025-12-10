"""
Video info display for Governor of Poker.

Shows video metadata and first-frame detection results.
"""

from .first_frame import TableInfo
from .frame_extractor import get_video_info


def format_info(video_path: str) -> str:
    """Format video and table information as a string."""
    info = get_video_info(video_path)
    table = TableInfo.from_video(video_path)

    lines = [
        f"Video: {info['path']}",
        f"  Resolution: {info['width']}x{info['height']}",
        f"  FPS: {info['fps']:.2f}",
        f"  Duration: {info['duration_seconds']:.1f}s ({info['duration_seconds']/60:.1f} min)",
        f"  Total frames: {info['frame_count']:,}",
        "",
    ]

    # Add table info
    for line in str(table).split('\n'):
        lines.append(line)

    return "\n".join(lines)
