"""
Command line interface for Lieutenant of Poker.

Usage:
    lieutenant extract-frames <video> [--output-dir=<dir>] [--interval=<ms>] [--format=<fmt>]
    lieutenant analyze <video> [--interval=<ms>] [--output=<file>]
    lieutenant export <video> [--format=<fmt>] [--output=<file>]
    lieutenant monitor [--window=<title>] [--fps=<n>] [--audio] [--overlay] [--log=<file>]
    lieutenant info <video>
    lieutenant --help
    lieutenant --version
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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
from lieutenant_of_poker.frame_extractor import VideoFrameExtractor
from lieutenant_of_poker.game_state import GameStateExtractor, GameState
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

    # calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="GUI tool for calibrating table regions"
    )
    calibrate_parser.add_argument("video", help="Path to video file")
    calibrate_parser.add_argument(
        "--frame", "-f", type=int, default=None,
        help="Frame number to use"
    )
    calibrate_parser.add_argument(
        "--timestamp", "-t", type=float, default=None,
        help="Timestamp in seconds"
    )
    calibrate_parser.add_argument(
        "--scale", "-s", type=float, default=None,
        help="Display scale factor (auto-calculated if not specified)"
    )

    # calibrate-hero command
    calibrate_hero_parser = subparsers.add_parser(
        "calibrate-hero", help="GUI tool for calibrating hero card subregions (rank/suit)"
    )
    calibrate_hero_parser.add_argument("video", help="Path to video file")
    calibrate_hero_parser.add_argument(
        "--frame", "-f", type=int, default=None,
        help="Frame number to use"
    )
    calibrate_hero_parser.add_argument(
        "--timestamp", "-t", type=float, default=None,
        help="Timestamp in seconds"
    )
    calibrate_hero_parser.add_argument(
        "--scale", "-s", type=float, default=4.0,
        help="Display scale factor (default: 4.0)"
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
        elif args.command == "calibrate":
            cmd_calibrate(args)
        elif args.command == "calibrate-hero":
            cmd_calibrate_hero(args)
        elif args.command == "monitor":
            cmd_monitor(args)
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
    import cv2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with VideoFrameExtractor(args.video) as video:
        start_ms = args.start * 1000
        end_ms = args.end * 1000 if args.end else video.duration_seconds * 1000

        # Calculate total frames for progress bar
        total_frames = int((end_ms - start_ms) / args.interval) + 1

        print(f"Extracting frames from {args.video}", file=sys.stderr)
        print(f"  Duration: {video.duration_seconds:.1f}s", file=sys.stderr)
        print(f"  Interval: {args.interval}ms", file=sys.stderr)
        print(f"  Expected frames: {total_frames}", file=sys.stderr)
        print(f"  Output: {output_dir}/", file=sys.stderr)

        count = 0
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

            for frame_info in video.iterate_at_interval(args.interval, start_ms, end_ms if args.end else None):
                timestamp_s = frame_info.timestamp_ms / 1000
                filename = f"frame_{timestamp_s:.2f}s.{args.format}"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame_info.image)
                count += 1
                progress.update(task, advance=1)

        print(f"Done! Extracted {count} frames to {output_dir}/", file=sys.stderr)


def cmd_analyze(args):
    """Analyze video and output game states."""
    extractor = GameStateExtractor()

    with VideoFrameExtractor(args.video) as video:
        start_ms = args.start * 1000
        end_ms = args.end * 1000 if args.end else video.duration_seconds * 1000

        # Calculate total frames for progress bar
        total_frames = int((end_ms - start_ms) / args.interval) + 1

        if args.verbose:
            print(f"Analyzing {args.video}", file=sys.stderr)
            print(f"  Duration: {video.duration_seconds:.1f}s", file=sys.stderr)
            print(f"  Expected frames: {total_frames}", file=sys.stderr)

        results = []

        # Use rich progress bar
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
            disable=not sys.stderr.isatty(),  # Disable if not a terminal
        ) as progress:
            task = progress.add_task("Analyzing frames", total=total_frames)

            for frame_info in video.iterate_at_interval(args.interval, start_ms, end_ms if args.end else None):
                state = extractor.extract(
                    frame_info.image,
                    frame_number=frame_info.frame_number,
                    timestamp_ms=frame_info.timestamp_ms
                )

                result = game_state_to_dict(state)
                results.append(result)
                progress.update(task, advance=1)

        print(f"Done! Analyzed {len(results)} frames.", file=sys.stderr)

        # Output results
        output = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output)


def cmd_export(args):
    """Export hand histories."""
    extractor = GameStateExtractor()
    exporter = HandHistoryExporter()

    with VideoFrameExtractor(args.video) as video:
        print(f"Analyzing {args.video} for hand export...", file=sys.stderr)

        # Calculate total frames for progress bar
        total_frames = int((video.duration_seconds * 1000) / args.interval) + 1

        # Collect all game states with progress bar
        states = []
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

            for frame_info in video.iterate_at_interval(args.interval):
                state = extractor.extract(
                    frame_info.image,
                    frame_number=frame_info.frame_number,
                    timestamp_ms=frame_info.timestamp_ms
                )
                states.append(state)
                progress.update(task, advance=1)

        print(f"Collected {len(states)} game states", file=sys.stderr)

        # For now, create a single hand from all states
        # (A more sophisticated implementation would detect hand boundaries)
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
    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {args.video}")
        print(f"  Resolution: {video.width}x{video.height}")
        print(f"  FPS: {video.fps:.2f}")
        print(f"  Duration: {video.duration_seconds:.1f}s ({video.duration_seconds/60:.1f} min)")
        print(f"  Total frames: {video.frame_count:,}")


def cmd_diagnose(args):
    """Generate detailed diagnostic report for a frame."""
    import webbrowser
    from lieutenant_of_poker.diagnostic import DiagnosticExtractor, generate_html_report

    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {args.video}", file=sys.stderr)
        print(f"  Resolution: {video.width}x{video.height}", file=sys.stderr)
        print(f"  Duration: {video.duration_seconds:.1f}s", file=sys.stderr)

        # Determine which frame to analyze
        if args.frame is not None:
            frame_info = video.get_frame_at(args.frame)
            if frame_info is None:
                raise ValueError(f"Could not read frame {args.frame}")
        elif args.timestamp is not None:
            frame_info = video.get_frame_at_timestamp(args.timestamp * 1000)
            if frame_info is None:
                raise ValueError(f"Could not read frame at timestamp {args.timestamp}s")
        else:
            # Default to first frame
            frame_info = video.get_frame_at(0)
            if frame_info is None:
                raise ValueError("Could not read first frame")

        print(f"\nAnalyzing frame {frame_info.frame_number} ({frame_info.timestamp_ms/1000:.2f}s)...", file=sys.stderr)

        # Run diagnostic extraction
        extractor = DiagnosticExtractor()
        report = extractor.extract_with_diagnostics(
            frame_info.image,
            frame_number=frame_info.frame_number,
            timestamp_ms=frame_info.timestamp_ms,
        )

        # Generate HTML report
        output_path = Path(args.output)
        html = generate_html_report(report, output_path)

        print(f"\nReport generated: {output_path}", file=sys.stderr)

        # Count successes/failures
        successes = sum(1 for s in report.steps if s.success)
        failures = sum(1 for s in report.steps if not s.success)
        print(f"  Steps: {successes} succeeded, {failures} failed", file=sys.stderr)

        # Open in browser if requested
        if args.open:
            webbrowser.open(f"file://{output_path.absolute()}")


def cmd_calibrate(args):
    """GUI tool for calibrating table regions."""
    from lieutenant_of_poker.calibrate import CalibrationTool

    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {args.video}", file=sys.stderr)
        print(f"Resolution: {video.width}x{video.height}", file=sys.stderr)

        if args.frame is not None:
            frame_info = video.get_frame_at(args.frame)
        elif args.timestamp is not None:
            frame_info = video.get_frame_at_timestamp(args.timestamp * 1000)
        else:
            frame_info = video.get_frame_at(0)

        if frame_info is None:
            raise ValueError("Could not read frame")

        frame = frame_info.image

        # Auto-calculate scale to fit on screen
        if args.scale is None:
            max_dim = max(video.width, video.height)
            scale = min(1.0, 1400 / max_dim)
        else:
            scale = args.scale

        print(f"Display scale: {scale:.2f}", file=sys.stderr)
        print(f"\nStarting calibration tool...", file=sys.stderr)
        print("Press 'h' in the window for help\n", file=sys.stderr)

        tool = CalibrationTool(frame, scale=scale)
        tool.run()


def cmd_calibrate_hero(args):
    """GUI tool for calibrating hero card subregions."""
    from lieutenant_of_poker.calibrate_hero import HeroCardCalibrationTool
    from lieutenant_of_poker.table_regions import detect_table_regions

    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {args.video}", file=sys.stderr)
        print(f"Resolution: {video.width}x{video.height}", file=sys.stderr)

        if args.frame is not None:
            frame_info = video.get_frame_at(args.frame)
        elif args.timestamp is not None:
            frame_info = video.get_frame_at_timestamp(args.timestamp * 1000)
        else:
            frame_info = video.get_frame_at(0)

        if frame_info is None:
            raise ValueError("Could not read frame")

        frame = frame_info.image

        # Extract the full hero cards region
        region_detector = detect_table_regions(frame)
        hero_region = region_detector.extract_hero_cards(frame)

        print(f"Hero region size: {hero_region.shape[1]}x{hero_region.shape[0]}", file=sys.stderr)
        print(f"\nStarting hero card calibration tool...", file=sys.stderr)

        tool = HeroCardCalibrationTool(hero_region, scale=args.scale)
        tool.run()


def cmd_monitor(args):
    """Live monitor the game with mistake detection."""
    from pathlib import Path

    from lieutenant_of_poker.screen_capture import (
        MacOSScreenCapture,
        check_screen_recording_permission,
        get_permission_instructions,
    )
    from lieutenant_of_poker.live_state_tracker import LiveStateTracker
    from lieutenant_of_poker.rules_engine import RulesEngine, Severity
    from lieutenant_of_poker.notifications import (
        NotificationManager,
        TerminalNotifier,
        AudioNotifier,
        LogNotifier,
        OverlayNotifier,
    )
    from lieutenant_of_poker.live_monitor import LiveMonitor
    from lieutenant_of_poker.rules import basic_rules

    # List rules mode
    if args.list_rules:
        rules = RulesEngine()
        basic_rules.register_all(rules)
        print("Available rules:")
        for name, enabled, description in rules.list_rules():
            status = "enabled" if enabled else "disabled"
            print(f"  {name}: {description} [{status}]")
        return

    # Check screen recording permission
    if not check_screen_recording_permission():
        print("Error: Screen recording permission not granted.", file=sys.stderr)
        print(get_permission_instructions(), file=sys.stderr)
        sys.exit(1)

    # Initialize screen capture
    try:
        capture = MacOSScreenCapture(window_title=args.window)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nMake sure '{args.window}' is open and visible.", file=sys.stderr)
        sys.exit(1)

    print(f"Found window: {capture.window}", file=sys.stderr)

    # Initialize state tracker
    tracker = LiveStateTracker()

    # Initialize rules engine
    rules = RulesEngine()
    basic_rules.register_all(rules)

    if args.rules:
        # Enable only specified rules
        rules.disable_all()
        for rule_name in args.rules:
            if not rules.enable_rule(rule_name):
                print(f"Warning: Unknown rule '{rule_name}'", file=sys.stderr)

    # Initialize notification channels
    notifications = NotificationManager()

    severity_map = {
        "info": Severity.INFO,
        "warning": Severity.WARNING,
        "error": Severity.ERROR,
        "critical": Severity.CRITICAL,
    }
    notifications.set_minimum_severity(severity_map[args.severity])

    if not args.quiet:
        term_notifier = TerminalNotifier()
        if term_notifier.is_available():
            notifications.add_channel(term_notifier)

    if args.audio:
        audio_notifier = AudioNotifier()
        if audio_notifier.is_available():
            notifications.add_channel(audio_notifier)
        else:
            print("Warning: Audio alerts not available", file=sys.stderr)

    if args.log:
        log_notifier = LogNotifier(Path(args.log))
        if log_notifier.is_available():
            notifications.add_channel(log_notifier)
            print(f"Logging to: {args.log}", file=sys.stderr)
        else:
            print(f"Warning: Cannot write to log file {args.log}", file=sys.stderr)

    if args.overlay:
        overlay_notifier = OverlayNotifier()
        if overlay_notifier.is_available():
            notifications.add_channel(overlay_notifier)
        else:
            print("Warning: Overlay notifications not available", file=sys.stderr)

    # Create and start monitor
    monitor = LiveMonitor(
        screen_capture=capture,
        state_tracker=tracker,
        rules_engine=rules,
        notification_manager=notifications,
        fps=args.fps,
    )

    print(f"\nStarting live monitor (Ctrl+C to stop)...", file=sys.stderr)
    print(f"  Window: {args.window}", file=sys.stderr)
    print(f"  FPS: {args.fps}", file=sys.stderr)
    print(f"  Severity: {args.severity}+", file=sys.stderr)
    print(f"  Channels: {notifications.channel_count}", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        monitor.start()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()

        stats = monitor.get_stats()
        print(f"\nSession Summary:", file=sys.stderr)
        print(f"  Duration: {stats.duration_seconds:.1f}s", file=sys.stderr)
        print(f"  Frames processed: {stats.frames_processed:,}", file=sys.stderr)
        print(f"  Hands tracked: {stats.hands_tracked}", file=sys.stderr)
        print(f"  Violations detected: {stats.violations_detected}", file=sys.stderr)
        print(f"  Avg frame time: {stats.avg_frame_time_ms:.1f}ms", file=sys.stderr)
        print(f"  Actual FPS: {stats.actual_fps:.1f}", file=sys.stderr)


def game_state_to_dict(state: GameState) -> dict:
    """Convert GameState to dictionary for JSON output."""
    return {
        "frame_number": state.frame_number,
        "timestamp_ms": state.timestamp_ms,
        "street": state.street.name,
        "pot": state.pot,
        "hero_chips": state.hero_chips,
        "community_cards": [str(c) for c in state.community_cards],
        "hero_cards": [str(c) for c in state.hero_cards],
        "players": {
            pos.name: {
                "chips": player.chips,
                "last_action": str(player.last_action) if player.last_action else None,
                "cards": [str(c) for c in player.cards],
            }
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
