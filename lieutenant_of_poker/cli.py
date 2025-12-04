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

from rich.progress import TextColumn

from lieutenant_of_poker.progress import create_progress

from lieutenant_of_poker import __version__
from lieutenant_of_poker.snowie_export import export_snowie
from lieutenant_of_poker.pokerstars_export import export_pokerstars
from lieutenant_of_poker.human_export import export_human
from lieutenant_of_poker.action_log_export import export_action_log


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
        "--start", "-s", type=float, default=0, help="Start timestamp in seconds (default: 0)"
    )
    extract_parser.add_argument(
        "--end", "-e", type=float, default=None, help="End timestamp in seconds (default: end of video)"
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
    analyze_parser.add_argument(
        "--table-background", "-b", default=None,
        help="Path to table background image for empty slot detection"
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
    batch_parser.add_argument(
        "--table-background", "-b", default=None,
        help="Path to table background image for empty slot detection"
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
        "--display", "-D", type=int, default=0,
        help="Display index to capture (default: 0 = main display)"
    )
    record_parser.add_argument(
        "--hotkey", "-H", type=str, default="cmd+shift+r",
        help="Global hotkey to toggle recording (default: cmd+shift+r)"
    )
    record_parser.add_argument(
        "--prefix", "-p", type=str, default="gop3",
        help="Filename prefix for recordings (default: gop3)"
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
    from lieutenant_of_poker.frame_extractor import extract_frames, get_video_info

    output_dir = Path(args.output_dir)
    info = get_video_info(args.video)

    start_ms = args.start * 1000
    end_ms = args.end * 1000 if args.end else info['duration_seconds'] * 1000

    # Calculate total frames based on video fps
    start_frame = int(start_ms * info['fps'] / 1000)
    end_frame = int(end_ms * info['fps'] / 1000) if args.end else info['frame_count']
    total_frames = end_frame - start_frame

    print(f"Extracting all frames from {args.video}", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)
    print(f"  FPS: {info['fps']:.1f}", file=sys.stderr)
    print(f"  Total frames: {total_frames:,}", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)

    with create_progress() as progress:
        task = progress.add_task("Extracting frames", total=total_frames)

        def on_progress(current, total):
            progress.update(task, completed=current)

        count = extract_frames(
            args.video,
            output_dir,
            format=args.format,
            start_ms=start_ms,
            end_ms=end_ms if args.end else None,
            on_progress=on_progress,
        )

    print(f"Done! Extracted {count:,} frames to {output_dir}/", file=sys.stderr)


def cmd_analyze(args):
    """Analyze video and output game states (every frame)."""
    from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig
    from lieutenant_of_poker.frame_extractor import get_video_info

    info = get_video_info(args.video)
    start_ms = args.start * 1000
    end_ms = args.end * 1000 if args.end else info['duration_seconds'] * 1000

    # Calculate total frames based on video fps
    start_frame = int(start_ms * info['fps'] / 1000)
    end_frame = int(end_ms * info['fps'] / 1000) if args.end else info['frame_count']
    total_frames = end_frame - start_frame

    # Detect player names and button from first frame
    from lieutenant_of_poker.first_frame import detect_from_video
    first = detect_from_video(args.video, start_ms)
    player_names = first.player_names
    print(first, file=sys.stderr)

    config = AnalysisConfig(
        start_ms=start_ms,
        end_ms=end_ms if args.end else None,
        table_background=args.table_background,
    )

    with create_progress(TextColumn("OCR: {task.fields[ocr]}")) as progress:
        task = progress.add_task("Analyzing frames", total=total_frames, ocr=0)

        def on_progress(p):
            progress.update(task, completed=p.current_frame, ocr=p.ocr_calls)

        states = analyze_video(
            args.video,
            config,
            on_progress=on_progress,
            include_rejected=args.verbose,
        )

    print(f"Done! Analyzed {len(states)} state changes.", file=sys.stderr)

    # Output results
    from .formatter import format_changes
    output = format_changes(states, verbose=args.verbose, player_names=player_names)
    print(output)


def cmd_export(args):
    """Export hand histories (analyzes every frame)."""
    from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig
    from lieutenant_of_poker.frame_extractor import get_video_info, VideoFrameExtractor
    from lieutenant_of_poker.snowie_export import export_snowie

    info = get_video_info(args.video)
    start_ms = args.start * 1000
    end_ms = args.end * 1000 if args.end else info['duration_seconds'] * 1000

    # Calculate total frames based on video fps
    start_frame = int(start_ms * info['fps'] / 1000)
    end_frame = int(end_ms * info['fps'] / 1000) if args.end else info['frame_count']
    total_frames = end_frame - start_frame

    # Detect button position and player names from first frames
    from lieutenant_of_poker.first_frame import detect_from_video
    first = detect_from_video(args.video, start_ms)
    button_pos = args.button if args.button is not None else (first.button_index if first.button_index is not None else 0)
    player_names = first.player_names

    config = AnalysisConfig(
        start_ms=start_ms,
        end_ms=end_ms if args.end else None,
    )

    with create_progress() as progress:
        task = progress.add_task("Extracting states", total=total_frames)

        def on_progress(p):
            progress.update(task, completed=p.current_frame)

        states = analyze_video(args.video, config, on_progress=on_progress)

    if not states:
        print("No hand data detected.", file=sys.stderr)
        return

    # Export based on format
    if args.format == "snowie":
        output = export_snowie(states, button_pos=button_pos, player_names=player_names)
    elif args.format == "human":
        output = export_human(states, button_pos=button_pos, player_names=player_names)
    elif args.format == "actions":
        output = export_action_log(states, button_pos=button_pos, player_names=player_names)
    else:  # pokerstars
        output = export_pokerstars(states, button_pos=button_pos, player_names=player_names)
        if not output:
            print("No hand data detected.", file=sys.stderr)
            return

    print(output)


def cmd_batch_export(args):
    """Export hand histories from all videos in a folder."""
    from .batch_export import batch_export
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else folder
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_export(
        folder,
        output_dir,
        args.format,
        args.extension,
        table_background=args.table_background,
    )


def cmd_info(args):
    """Show video information."""
    from lieutenant_of_poker.frame_extractor import get_video_info

    info = get_video_info(args.video)
    print(f"Video: {info['path']}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration_seconds']:.1f}s ({info['duration_seconds']/60:.1f} min)")
    print(f"  Total frames: {info['frame_count']:,}")

    if args.players:
        from lieutenant_of_poker.first_frame import detect_from_video
        first = detect_from_video(args.video)
        for line in str(first).split('\n'):
            print(f"  {line}")


def cmd_diagnose(args):
    """Generate detailed diagnostic report for a frame."""
    import webbrowser
    from lieutenant_of_poker.diagnostic import generate_diagnostic_report
    from lieutenant_of_poker.frame_extractor import get_video_info

    info = get_video_info(args.video)
    print(f"Video: {args.video}", file=sys.stderr)
    print(f"  Resolution: {info['width']}x{info['height']}", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)

    output_path = Path(args.output)
    result = generate_diagnostic_report(
        args.video,
        output_path,
        frame_number=args.frame,
        timestamp_s=args.timestamp,
    )

    print(f"\nAnalyzing frame {result['frame_number']} ({result['timestamp_ms']/1000:.2f}s)...", file=sys.stderr)
    print(f"Report generated: {output_path}", file=sys.stderr)
    print(f"  Steps: {result['steps_succeeded']} succeeded, {result['steps_failed']} failed", file=sys.stderr)

    if args.open:
        webbrowser.open(f"file://{output_path.absolute()}")


def cmd_clear_library(args):
    """Clear card reference image libraries."""
    from lieutenant_of_poker.card_matcher import LIBRARY_DIR as CARD_LIBRARY_DIR

    count = 0
    for png_file in CARD_LIBRARY_DIR.rglob("*.png"):
        png_file.unlink()
        count += 1

    print(f"Cleared {count} card library images", file=sys.stderr)


def cmd_record(args):
    """Simple screen recording with hotkey toggle."""
    from lieutenant_of_poker.screen_capture import (
        ScreenCaptureKitCapture,
        check_screen_recording_permission,
        get_permission_instructions,
    )
    from lieutenant_of_poker.video_recorder import RecordingSession
    from lieutenant_of_poker.hotkeys import create_hotkey_listener, play_sound
    from lieutenant_of_poker.notifications import OverlayNotifier

    # Check screen recording permission
    if not check_screen_recording_permission():
        print("Error: Screen recording permission not granted.", file=sys.stderr)
        print(get_permission_instructions(), file=sys.stderr)
        sys.exit(1)

    # Initialize screen capture (always use ScreenCaptureKit for fullscreen support)
    try:
        capture = ScreenCaptureKitCapture(
            capture_display=True,
            display_id=args.display,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Capture target: {capture.window}", file=sys.stderr)

    # Set up overlay notifier
    overlay = OverlayNotifier()

    # Callback for recording state changes (used by both hotkey and auto-detect)
    def on_recording_change(is_recording: bool, path):
        if is_recording:
            print(f"\n🔴 Recording started: {path}", file=sys.stderr)
            play_sound("Blow")
            overlay.send_message("Recording Started", str(Path(path).name))
        else:
            print(f"\n⬜ Recording stopped: {path}", file=sys.stderr)
            play_sound("Glass")
            overlay.send_message("Recording Stopped", str(Path(path).name))

    # Callback for frame drops
    def on_frame_drop(actual_fps: float, target_fps: float):
        pct = (actual_fps / target_fps) * 100
        print(f"\n⚠️  Frame drop detected: {actual_fps:.1f}/{target_fps} FPS ({pct:.0f}%)", file=sys.stderr)
        play_sound("Basso")  # Warning sound
        overlay.send_message("⚠️ Dropping Frames", f"{actual_fps:.1f} FPS (target: {target_fps})")

    # Callback for debug stats
    def on_debug_stats(stats: dict):
        rec_indicator = "REC" if stats["is_recording"] else "---"
        fps_pct = stats["fps_ratio"] * 100
        cap = stats["capture_ms"]
        det = stats["detect_ms"]
        wrt = stats["write_ms"]
        loop = stats["loop_ms"]
        drift = stats["sleep_drift_ms"]

        print(f"\n[DEBUG {rec_indicator}] FPS: {stats['actual_fps']:.1f}/{stats['target_fps']} ({fps_pct:.0f}%) | "
              f"frames: {stats['total_frames']} | overruns: {stats['overruns']}", file=sys.stderr)
        print(f"  capture: {cap['avg']:.1f}ms avg, {cap['max']:.1f}ms max", file=sys.stderr)
        if det["count"] > 0:
            print(f"  detect:  {det['avg']:.1f}ms avg, {det['max']:.1f}ms max", file=sys.stderr)
        if wrt["count"] > 0:
            print(f"  write:   {wrt['avg']:.1f}ms avg, {wrt['max']:.1f}ms max", file=sys.stderr)
        print(f"  loop:    {loop['avg']:.1f}ms avg, {loop['max']:.1f}ms max (target: {1000/args.fps:.1f}ms)", file=sys.stderr)
        print(f"  drift:   {drift['avg']:.1f}ms avg, {drift['max']:.1f}ms max", file=sys.stderr)

    # Create recording session
    session = RecordingSession(
        capture=capture,
        output_dir=args.output_dir,
        fps=args.fps,
        file_prefix=args.prefix,
        auto_detect=args.auto,
        on_recording_change=on_recording_change if args.auto else None,
        on_frame_drop=on_frame_drop,
        debug=getattr(args, 'debug', False),
        on_debug_stats=on_debug_stats if getattr(args, 'debug', False) else None,
    )

    # Set up hotkey for manual toggle
    def on_hotkey():
        is_recording, stopped_path = session.toggle_recording()
        on_recording_change(is_recording, session._recorder.current_path if is_recording else stopped_path)

    hotkey_listener = create_hotkey_listener(args.hotkey, on_hotkey)

    print(f"\nReady to record (Ctrl+C to exit)...", file=sys.stderr)
    print(f"  Display: {args.display}", file=sys.stderr)
    print(f"  FPS: {args.fps}", file=sys.stderr)
    print(f"  Output: {Path(args.output_dir).absolute()}", file=sys.stderr)
    print(f"  Hotkey: {args.hotkey} (toggle recording)", file=sys.stderr)
    if args.auto:
        if session.auto_detect_available:
            print(f"  Auto-detect: ENABLED (mask loaded)", file=sys.stderr)
        else:
            print(f"  Auto-detect: FAILED (mask.png not found)", file=sys.stderr)
    if getattr(args, 'debug', False):
        print(f"  Debug: ENABLED (stats every 5s)", file=sys.stderr)
    print(f"\nPress {args.hotkey} to start/stop recording...", file=sys.stderr)

    try:
        session.start()
    except KeyboardInterrupt:
        pass
    finally:
        if hotkey_listener:
            hotkey_listener.stop()
        session.stop()

        print(f"\nSession ended.", file=sys.stderr)
        if session.recordings:
            print(f"Recordings saved ({len(session.recordings)}):", file=sys.stderr)
            for rec in session.recordings:
                print(f"  - {rec}", file=sys.stderr)
        else:
            print("No recordings made.", file=sys.stderr)


def cmd_split(args):
    """Split video into chunks based on brightness detection."""
    from lieutenant_of_poker.video_splitter import detect_segments, split_video

    video_path = Path(args.video)
    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent
    prefix = args.prefix if args.prefix else video_path.stem

    from lieutenant_of_poker.frame_extractor import get_video_info
    info = get_video_info(args.video)
    total_frames = info['frame_count']

    print(f"Analyzing {video_path}...", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)
    print(f"  Threshold: {args.threshold}, Consecutive: {args.consecutive}", file=sys.stderr)

    with create_progress() as progress:
        task = progress.add_task("Scanning frames", total=total_frames)

        def on_progress(current, total):
            progress.update(task, completed=current)

        segments = detect_segments(
            video_path,
            threshold=args.threshold,
            consecutive_frames=args.consecutive,
            step=args.step,
            on_progress=on_progress,
        )

    # Filter and display
    min_ms = args.min_duration * 1000
    filtered = [s for s in segments if s.duration_ms >= min_ms]

    print(f"\nDetected {len(segments)} segments ({len(filtered)} >= {args.min_duration}s):", file=sys.stderr)
    for i, seg in enumerate(filtered):
        print(f"  {i+1}. {seg.start_ms/1000:.2f}s - {seg.end_ms/1000:.2f}s ({seg.duration_s:.1f}s)", file=sys.stderr)

    if not filtered:
        print("\nNo segments found.", file=sys.stderr)
        return

    if args.dry_run:
        print("\nDry run - no files created.", file=sys.stderr)
        return

    print(f"\nSplitting into {len(filtered)} chunks...", file=sys.stderr)

    def on_chunk(num, total, path):
        print(f"  Creating {path.name}...", file=sys.stderr, end=" ", flush=True)

    result = split_video(video_path, segments, output_dir, prefix, args.min_duration, on_chunk)

    for path in result.created_files:
        print("OK", file=sys.stderr)
    for path, error in result.failed_files:
        print(f"FAILED: {error}", file=sys.stderr)

    print(f"\nCreated {len(result.created_files)} chunk(s) in {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
