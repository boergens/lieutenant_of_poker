"""
Command line interface for Lieutenant of Poker.

Usage:
    lieutenant record [--fps=<n>] [--display=<n>] [--hotkey=<key>]
    lieutenant analyze <video>
    lieutenant extract-frames <video> [--output-dir=<dir>]
    lieutenant export <video> [--format=<fmt>] [--output=<file>]
    lieutenant info <video>
    lieutenant --help
    lieutenant --version
"""

import argparse
import sys
from pathlib import Path

from lieutenant_of_poker import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="lieutenant",
        description="Lieutenant of Poker - Video analysis for Governor of Poker",
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract-frames command
    extract_parser = subparsers.add_parser(
        "extract-frames", help="Extract all frames from a video file"
    )
    extract_parser.add_argument("video", help="Path to video file")
    extract_parser.add_argument(
        "--output-dir", "-o", default="frames", help="Output directory (default: frames)"
    )
    extract_parser.add_argument(
        "--format", "-f", default="jpg", choices=["jpg", "png"], help="Output format (default: jpg)"
    )
    extract_parser.add_argument(
        "--start-ms", "-s", type=float, default=0, help="Start timestamp in milliseconds (default: 0)"
    )
    extract_parser.add_argument(
        "--end-ms", "-e", type=float, default=None, help="End timestamp in milliseconds (default: end of video)"
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze video and extract game states (every frame)"
    )
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument(
        "--start", "-s", type=float, default=0, help="Start timestamp in seconds (default: 0)"
    )
    analyze_parser.add_argument(
        "--end", "-e", type=float, default=None, help="End timestamp in seconds (default: end of video)"
    )
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Output verbose per-frame extractions (no filtering/consensus)"
    )

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export hand histories from video (analyzes every frame)"
    )
    export_parser.add_argument("video", help="Path to video file")
    export_parser.add_argument(
        "--format", "-f", default="actions", choices=["pokerstars", "snowie", "human", "actions"],
        help="Output format (default: actions)"
    )
    export_parser.add_argument(
        "--start", "-s", type=float, default=0, help="Start timestamp in seconds (default: 0)"
    )
    export_parser.add_argument(
        "--end", "-e", type=float, default=None, help="End timestamp in seconds (default: end of video)"
    )
    export_parser.add_argument(
        "--button", "-b", type=int, default=None,
        help="Button position (0=SEAT_1, 1=SEAT_2, 2=SEAT_3, 3=SEAT_4, 4=hero). Auto-detected if not specified."
    )

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show video information"
    )
    info_parser.add_argument("video", help="Path to video file")
    info_parser.add_argument(
        "--players", "-p", action="store_true",
        help="Detect and display player names from first frame"
    )

    # diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Generate detailed diagnostic report for a frame"
    )
    diagnose_parser.add_argument("video", help="Path to video file")
    diagnose_parser.add_argument(
        "--frame", "-f", type=int, default=None,
        help="Frame number to analyze"
    )
    diagnose_parser.add_argument(
        "--timestamp", "-t", type=float, default=None,
        help="Timestamp in seconds to analyze"
    )
    diagnose_parser.add_argument(
        "--output", "-o", default="diagnostic_report.html",
        help="Output HTML file (default: diagnostic_report.html)"
    )
    diagnose_parser.add_argument(
        "--open", action="store_true",
        help="Open the report in browser after generation"
    )

    # clear-library command
    clear_parser = subparsers.add_parser(
        "clear-library", help="Clear all cached reference images from card libraries"
    )

    # split command - split video by brightness detection
    split_parser = subparsers.add_parser(
        "split", help="Split video into chunks based on screen on/off detection"
    )
    split_parser.add_argument("video", help="Path to video file")
    split_parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory for chunks (default: same as input video)"
    )
    split_parser.add_argument(
        "--prefix", "-p", type=str, default=None,
        help="Filename prefix for chunks (default: input filename)"
    )
    split_parser.add_argument(
        "--threshold", "-t", type=float, default=250.0,
        help="Brightness threshold 0-255 (default: 250)"
    )
    split_parser.add_argument(
        "--consecutive", "-c", type=int, default=3,
        help="Consecutive frames needed to trigger (default: 3)"
    )
    split_parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show detected segments without creating files"
    )
    split_parser.add_argument(
        "--min-duration", "-m", type=float, default=1.0,
        help="Minimum segment duration in seconds (default: 1.0)"
    )
    split_parser.add_argument(
        "--step", "-s", type=int, default=2,
        help="Process every Nth frame for detection (default: 2, use 1 for all frames)"
    )

    # batch-export command
    batch_parser = subparsers.add_parser(
        "batch-export", help="Export hand histories from all videos in a folder"
    )
    batch_parser.add_argument("folder", help="Path to folder containing video files")
    batch_parser.add_argument(
        "--format", "-f", default="snowie", choices=["pokerstars", "snowie", "human", "actions"],
        help="Output format (default: snowie)"
    )
    batch_parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory for text files (default: same as input folder)"
    )
    batch_parser.add_argument(
        "--extension", "-e", default=".txt",
        help="Output file extension (default: .txt)"
    )

    # record command - simple screen recording
    record_parser = subparsers.add_parser(
        "record", help="Record screen to video (simple, no analysis)"
    )
    record_parser.add_argument(
        "--output-dir", "-o", type=str, default=".",
        help="Directory for recorded videos (default: current directory)"
    )
    record_parser.add_argument(
        "--fps", "-f", type=int, default=10,
        help="Frames per second (default: 10)"
    )
    record_parser.add_argument(
        "--hotkey", "-H", type=str, default="cmd+shift+r",
        help="Global hotkey to toggle recording (default: cmd+shift+r)"
    )
    record_parser.add_argument(
        "--prefix", "-p", type=str, default="poker",
        help="Filename prefix for recordings (default: poker)"
    )
    record_parser.add_argument(
        "--auto", "-a", action="store_true",
        help="Auto-detect recording start/stop based on mask brightness"
    )
    record_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output with timing statistics"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "extract-frames":
            cmd_extract_frames(args)
        elif args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "export":
            cmd_export(args)
        elif args.command == "info":
            cmd_info(args)
        elif args.command == "diagnose":
            cmd_diagnose(args)
        elif args.command == "record":
            cmd_record(args)
        elif args.command == "split":
            cmd_split(args)
        elif args.command == "batch-export":
            cmd_batch_export(args)
        elif args.command == "clear-library":
            cmd_clear_library(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


def cmd_extract_frames(args):
    """Extract all frames from video."""
    from lieutenant_of_poker.frame_extractor import extract_frames

    extract_frames(args.video, Path(args.output_dir), args.format, args.start_ms, args.end_ms)


def cmd_analyze(args):
    """Analyze video and output game states (every frame)."""
    from lieutenant_of_poker.analysis import analyze_and_format

    output = analyze_and_format(args.video, args.start, args.end, args.verbose)
    print(output)


def cmd_export(args):
    """Export hand histories (analyzes every frame)."""
    from lieutenant_of_poker.analysis import analyze_and_export

    output = analyze_and_export(args.video, args.format, args.start, args.end, args.button)
    if output:
        print(output)


def cmd_batch_export(args):
    """Export hand histories from all videos in a folder."""
    from lieutenant_of_poker.batch_export import batch_export

    batch_export(args.folder, args.output_dir, args.format, args.extension)


def cmd_info(args):
    """Show video information."""
    from lieutenant_of_poker.frame_extractor import format_video_info

    print(format_video_info(args.video, args.players))


def cmd_diagnose(args):
    """Generate detailed diagnostic report for a frame."""
    from lieutenant_of_poker.diagnostic import diagnose

    diagnose(args.video, args.output, args.frame, args.timestamp, args.open)


def cmd_clear_library(args):
    """Clear card reference image libraries."""
    from lieutenant_of_poker.card_matcher import clear_library

    clear_library()


def cmd_record(args):
    """Simple screen recording with hotkey toggle."""
    from lieutenant_of_poker.video_recorder import run_recording_session

    run_recording_session(args.output_dir, args.fps, args.hotkey, args.prefix, args.auto, args.debug)


def cmd_split(args):
    """Split video into chunks based on brightness detection."""
    from lieutenant_of_poker.video_splitter import split_video_by_brightness

    split_video_by_brightness(
        args.video, args.output_dir, args.prefix, args.threshold,
        args.consecutive, args.min_duration, args.step, args.dry_run,
    )


if __name__ == "__main__":
    main()
