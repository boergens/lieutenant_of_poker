"""
Command line interface for Lieutenant of Poker.

Usage:
    lieutenant record [--fps=<n>] [--display=<n>] [--hotkey=<key>]
    lieutenant analyze <video> [--interval=<ms>] [--output=<file>]
    lieutenant monitor [--window=<title>] [--fps=<n>] [--fullscreen]
    lieutenant extract-frames <video> [--output-dir=<dir>] [--interval=<ms>]
    lieutenant export <video> [--format=<fmt>] [--output=<file>]
    lieutenant info <video>
    lieutenant --help
    lieutenant --version
"""

import argparse
import json
import sys
from pathlib import Path

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

from lieutenant_of_poker import __version__
from lieutenant_of_poker.game_state import GameState
from lieutenant_of_poker.hand_history import HandHistoryExporter


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
        "extract-frames", help="Extract frames from a video file"
    )
    extract_parser.add_argument("video", help="Path to video file")
    extract_parser.add_argument(
        "--output-dir", "-o", default="frames", help="Output directory (default: frames)"
    )
    extract_parser.add_argument(
        "--interval", "-i", type=int, default=1000, help="Interval between frames in ms (default: 1000)"
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
        "analyze", help="Analyze video and extract game states"
    )
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument(
        "--interval", "-i", type=int, default=1000, help="Interval between frames in ms (default: 1000)"
    )
    analyze_parser.add_argument(
        "--output", "-o", default=None, help="Output file for JSON results (default: stdout)"
    )
    analyze_parser.add_argument(
        "--start", "-s", type=float, default=0, help="Start timestamp in seconds (default: 0)"
    )
    analyze_parser.add_argument(
        "--end", "-e", type=float, default=None, help="End timestamp in seconds (default: end of video)"
    )
    analyze_parser.add_argument(
        "--verbose", "-V", action="store_true", help="Print progress to stderr"
    )
    analyze_parser.add_argument(
        "--json", "-j", action="store_true", help="Output raw JSON instead of changes"
    )
    analyze_parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Generate diagnostic report for frames with unmatched images"
    )
    analyze_parser.add_argument(
        "--debug-dir", default="debug_frames",
        help="Directory for debug diagnostic reports (default: debug_frames)"
    )

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export hand histories from video"
    )
    export_parser.add_argument("video", help="Path to video file")
    export_parser.add_argument(
        "--format", "-f", default="pokerstars", choices=["pokerstars", "json"],
        help="Output format (default: pokerstars)"
    )
    export_parser.add_argument(
        "--output", "-o", default=None, help="Output file (default: stdout)"
    )
    export_parser.add_argument(
        "--interval", "-i", type=int, default=500, help="Analysis interval in ms (default: 500)"
    )

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show video information"
    )
    info_parser.add_argument("video", help="Path to video file")

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

    # monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Live monitor the game with mistake detection"
    )
    monitor_parser.add_argument(
        "--window", "-w", default="Governor of Poker",
        help="Window title to capture (default: Governor of Poker)"
    )
    monitor_parser.add_argument(
        "--fps", "-f", type=int, default=10,
        help="Frames per second to capture (default: 10)"
    )
    monitor_parser.add_argument(
        "--audio", "-a", action="store_true",
        help="Enable audio alerts"
    )
    monitor_parser.add_argument(
        "--overlay", action="store_true",
        help="Show visual overlay notifications (macOS only)"
    )
    monitor_parser.add_argument(
        "--log", "-l", type=str, default=None,
        help="Log file path for alerts"
    )
    monitor_parser.add_argument(
        "--rules", "-r", nargs="*", default=None,
        help="Specific rules to enable (default: all)"
    )
    monitor_parser.add_argument(
        "--severity", "-s",
        choices=["info", "warning", "error", "critical"],
        default="warning",
        help="Minimum severity to alert on (default: warning)"
    )
    monitor_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress terminal output (useful with --log)"
    )
    monitor_parser.add_argument(
        "--list-rules", action="store_true",
        help="List available rules and exit"
    )
    monitor_parser.add_argument(
        "--list-windows", action="store_true",
        help="List available windows and exit"
    )
    monitor_parser.add_argument(
        "--record", "-R", type=str, default=None,
        help="Record session to video file (e.g., session.mp4)"
    )
    monitor_parser.add_argument(
        "--fullscreen", "-F", action="store_true",
        help="Use ScreenCaptureKit for fullscreen app capture (macOS 12.3+)"
    )
    monitor_parser.add_argument(
        "--display", "-D", type=int, default=0,
        help="Display index to capture when using --fullscreen (default: 0)"
    )
    monitor_parser.add_argument(
        "--hotkey", "-H", type=str, default="cmd+shift+r",
        help="Global hotkey to toggle recording (default: cmd+shift+r)"
    )
    monitor_parser.add_argument(
        "--output-dir", "-O", type=str, default=".",
        help="Directory for recorded videos (default: current directory)"
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
        elif args.command == "monitor":
            cmd_monitor(args)
        elif args.command == "record":
            cmd_record(args)
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
    """Extract frames from video."""
    from lieutenant_of_poker.analysis import extract_frames, get_video_info

    output_dir = Path(args.output_dir)
    info = get_video_info(args.video)

    start_ms = args.start * 1000
    end_ms = args.end * 1000 if args.end else info['duration_seconds'] * 1000
    total_frames = int((end_ms - start_ms) / args.interval) + 1

    print(f"Extracting frames from {args.video}", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)
    print(f"  Interval: {args.interval}ms", file=sys.stderr)
    print(f"  Expected frames: {total_frames}", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=None,
        transient=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        task = progress.add_task("Extracting frames", total=total_frames)

        def on_progress(current, total):
            progress.update(task, completed=current)

        count = extract_frames(
            args.video,
            output_dir,
            interval_ms=args.interval,
            format=args.format,
            start_ms=start_ms,
            end_ms=end_ms if args.end else None,
            on_progress=on_progress,
        )

    print(f"Done! Extracted {count} frames to {output_dir}/", file=sys.stderr)


def cmd_analyze(args):
    """Analyze video and output game states."""
    from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig, get_video_info

    info = get_video_info(args.video)
    start_ms = args.start * 1000
    end_ms = args.end * 1000 if args.end else info['duration_seconds'] * 1000
    total_frames = int((end_ms - start_ms) / args.interval) + 1

    if args.verbose:
        print(f"Analyzing {args.video}", file=sys.stderr)
        print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)
        print(f"  Expected frames: {total_frames}", file=sys.stderr)

    # Setup debug mode
    debug_dir = None
    debug_count = 0
    diagnostic_extractor = None
    if args.debug:
        from lieutenant_of_poker.diagnostic import DiagnosticExtractor, generate_html_report
        from lieutenant_of_poker.fast_ocr import enable_ocr_debug
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_extractor = DiagnosticExtractor()
        # Enable OCR debug image saving
        ocr_debug_dir = debug_dir / "ocr_images"
        enable_ocr_debug(ocr_debug_dir)
        print(f"Debug mode: saving diagnostics to {debug_dir}/", file=sys.stderr)
        print(f"  OCR images: {ocr_debug_dir}/", file=sys.stderr)

    config = AnalysisConfig(
        interval_ms=args.interval,
        start_ms=start_ms,
        end_ms=end_ms if args.end else None,
        debug_dir=debug_dir,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("OCR: {task.fields[ocr]}"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=None,
        transient=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        task = progress.add_task("Analyzing frames", total=total_frames, ocr=0)

        def on_progress(p):
            progress.update(task, completed=p.current_frame, ocr=p.ocr_calls)

        def on_debug_frame(frame_info, state, reason):
            nonlocal debug_count
            debug_count += 1
            report = diagnostic_extractor.extract_with_diagnostics(
                frame_info.image,
                frame_number=frame_info.frame_number,
                timestamp_ms=frame_info.timestamp_ms,
            )
            report_path = debug_dir / f"frame_{frame_info.frame_number}_{frame_info.timestamp_ms:.0f}ms_{reason}.html"
            generate_html_report(report, report_path)

        states = analyze_video(
            args.video,
            config,
            on_progress=on_progress,
            on_debug_frame=on_debug_frame if args.debug else None,
        )

    print(f"Done! Analyzed {len(states)} frames.", file=sys.stderr)
    if args.debug:
        from lieutenant_of_poker.fast_ocr import disable_ocr_debug
        disable_ocr_debug()
        ocr_dir = debug_dir / "ocr_images"
        pot_count = len(list((ocr_dir / "pot").glob("*.png"))) if (ocr_dir / "pot").exists() else 0
        player_count = len(list((ocr_dir / "player").glob("*.png"))) if (ocr_dir / "player").exists() else 0
        print(f"Generated {debug_count} debug diagnostic reports in {debug_dir}/", file=sys.stderr)
        print(f"Saved OCR images: {pot_count} pot, {player_count} player in {ocr_dir}/", file=sys.stderr)

    # Output results
    if args.json:
        results = [game_state_to_dict(s) for s in states]
        output = json.dumps(results, indent=2)
    else:
        from .formatter import format_changes
        output = format_changes(states)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


def cmd_export(args):
    """Export hand histories."""
    from lieutenant_of_poker.analysis import analyze_video, AnalysisConfig, get_video_info

    exporter = HandHistoryExporter()
    info = get_video_info(args.video)
    total_frames = int((info['duration_seconds'] * 1000) / args.interval) + 1

    print(f"Analyzing {args.video} for hand export...", file=sys.stderr)

    config = AnalysisConfig(interval_ms=args.interval)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=None,
        transient=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        task = progress.add_task("Extracting states", total=total_frames)

        def on_progress(p):
            progress.update(task, completed=p.current_frame)

        states = analyze_video(args.video, config, on_progress=on_progress)

    print(f"Collected {len(states)} game states", file=sys.stderr)

    # Create hand from states
    hand = exporter.create_hand_from_states(states)

    if hand is None:
        print("No hand data detected.", file=sys.stderr)
        return

    # Export
    if args.format == "pokerstars":
        output = exporter.export_pokerstars_format(hand)
    else:
        output = json.dumps(hand_to_dict(hand), indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Hand history written to {args.output}", file=sys.stderr)
    else:
        print(output)


def cmd_info(args):
    """Show video information."""
    from lieutenant_of_poker.analysis import get_video_info

    info = get_video_info(args.video)
    print(f"Video: {info['path']}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration_seconds']:.1f}s ({info['duration_seconds']/60:.1f} min)")
    print(f"  Total frames: {info['frame_count']:,}")


def cmd_diagnose(args):
    """Generate detailed diagnostic report for a frame."""
    import webbrowser
    from lieutenant_of_poker.analysis import generate_diagnostic_report, get_video_info

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

    # Create recording session
    session = RecordingSession(
        capture=capture,
        output_dir=args.output_dir,
        fps=args.fps,
        file_prefix=args.prefix,
        auto_detect=args.auto,
        on_recording_change=on_recording_change if args.auto else None,
        on_frame_drop=on_frame_drop,
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


def cmd_monitor(args):
    """Live monitor the game with mistake detection."""
    from datetime import datetime
    from lieutenant_of_poker.screen_capture import check_screen_recording_permission, get_permission_instructions
    from lieutenant_of_poker.live_monitor import (
        MonitorConfig, create_live_monitor, list_available_windows, list_available_rules
    )
    from lieutenant_of_poker.hotkeys import create_hotkey_listener, play_sound
    from lieutenant_of_poker.notifications import OverlayNotifier

    # List rules mode
    if args.list_rules:
        print("Available rules:")
        for name, enabled, description in list_available_rules():
            status = "enabled" if enabled else "disabled"
            print(f"  {name}: {description} [{status}]")
        return

    # List windows mode
    if args.list_windows:
        print("Available windows:")
        for w in list_available_windows():
            print(f"  [{w['window_id']}] {w['owner']}: {w['title']} ({w['width']}x{w['height']})")
        return

    # Check screen recording permission
    if not check_screen_recording_permission():
        print("Error: Screen recording permission not granted.", file=sys.stderr)
        print(get_permission_instructions(), file=sys.stderr)
        sys.exit(1)

    # Create config from args
    config = MonitorConfig(
        window_title=args.window,
        fullscreen=args.fullscreen,
        display_id=args.display,
        fps=args.fps,
        enabled_rules=args.rules if args.rules else None,
        min_severity=args.severity,
        terminal_output=not args.quiet,
        audio_alerts=args.audio,
        overlay=args.overlay,
        log_file=Path(args.log) if args.log else None,
        record_to=Path(args.record) if args.record else None,
    )

    # Create monitor
    try:
        monitor = create_live_monitor(config)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Capture target: {monitor.capture.window}", file=sys.stderr)

    # Set up hotkey for recording toggle
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings_made = []
    overlay = OverlayNotifier()

    def toggle_recording():
        if monitor.is_recording:
            path = monitor.stop_recording()
            if path:
                recordings_made.append(path)
                print(f"\n⬜ Recording stopped: {path}", file=sys.stderr)
                play_sound("Glass")
                overlay.send_message("Recording Stopped", path.name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"gop3_{timestamp}.mp4"
            monitor.start_recording(filename)
            print(f"\n🔴 Recording started: {filename}", file=sys.stderr)
            play_sound("Blow")
            overlay.send_message("Recording Started", filename.name)

    hotkey_listener = create_hotkey_listener(args.hotkey, toggle_recording)

    # Print startup info
    print(f"\nStarting live monitor (Ctrl+C to stop)...", file=sys.stderr)
    print(f"  Mode: {'Fullscreen' if args.fullscreen else 'Window'}", file=sys.stderr)
    print(f"  FPS: {args.fps}", file=sys.stderr)
    print(f"  Severity: {args.severity}+", file=sys.stderr)
    print(f"  Recording hotkey: {args.hotkey}", file=sys.stderr)
    print(f"  Output directory: {output_dir.absolute()}", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        monitor.start()
    except KeyboardInterrupt:
        pass
    finally:
        if hotkey_listener:
            hotkey_listener.stop()
        monitor.stop()

        stats = monitor.get_stats()
        print(f"\nSession Summary:", file=sys.stderr)
        print(f"  Duration: {stats.duration_seconds:.1f}s", file=sys.stderr)
        print(f"  Frames processed: {stats.frames_processed:,}", file=sys.stderr)
        print(f"  Hands tracked: {stats.hands_tracked}", file=sys.stderr)
        print(f"  Violations detected: {stats.violations_detected}", file=sys.stderr)
        print(f"  Avg frame time: {stats.avg_frame_time_ms:.1f}ms", file=sys.stderr)
        print(f"  Actual FPS: {stats.actual_fps:.1f}", file=sys.stderr)
        if recordings_made:
            print(f"  Recordings: {len(recordings_made)}", file=sys.stderr)
            for rec in recordings_made:
                print(f"    - {rec}", file=sys.stderr)


def game_state_to_dict(state: GameState) -> dict:
    """Convert GameState to dictionary for JSON output."""
    return {
        "frame_number": state.frame_number,
        "timestamp_ms": state.timestamp_ms,
        "street": state.street.name,
        "pot": state.pot,
        "community_cards": [str(c) for c in state.community_cards],
        "hero_cards": [str(c) for c in state.hero_cards],
        "players": {
            pos.name: player.chips
            for pos, player in state.players.items()
        }
    }


def hand_to_dict(hand) -> dict:
    """Convert HandHistory to dictionary for JSON output."""
    return {
        "hand_id": hand.hand_id,
        "table_name": hand.table_name,
        "timestamp": hand.timestamp.isoformat(),
        "small_blind": hand.small_blind,
        "big_blind": hand.big_blind,
        "pot": hand.pot,
        "hero_cards": [c.short_name for c in hand.hero_cards],
        "flop_cards": [c.short_name for c in hand.flop_cards],
        "turn_card": hand.turn_card.short_name if hand.turn_card else None,
        "river_card": hand.river_card.short_name if hand.river_card else None,
        "players": [(name, chips, str(pos)) for name, chips, pos in hand.players],
    }


if __name__ == "__main__":
    main()
