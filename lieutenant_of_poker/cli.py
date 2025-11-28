"""
Command line interface for Lieutenant of Poker.

Usage:
    lieutenant extract-frames <video> [--output-dir=<dir>] [--interval=<ms>] [--format=<fmt>]
    lieutenant analyze <video> [--interval=<ms>] [--output=<file>]
    lieutenant export <video> [--format=<fmt>] [--output=<file>]
    lieutenant info <video>
    lieutenant --help
    lieutenant --version
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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
    except FileNotFoundError as e:
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
        end_ms = args.end * 1000 if args.end else None

        print(f"Extracting frames from {args.video}", file=sys.stderr)
        print(f"  Duration: {video.duration_seconds:.1f}s", file=sys.stderr)
        print(f"  Interval: {args.interval}ms", file=sys.stderr)
        print(f"  Output: {output_dir}/", file=sys.stderr)

        count = 0
        for frame_info in video.iterate_at_interval(args.interval, start_ms, end_ms):
            timestamp_s = frame_info.timestamp_ms / 1000
            filename = f"frame_{timestamp_s:.2f}s.{args.format}"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame_info.image)
            count += 1

            if count % 10 == 0:
                print(f"  Extracted {count} frames...", file=sys.stderr)

        print(f"Done! Extracted {count} frames to {output_dir}/", file=sys.stderr)


def cmd_analyze(args):
    """Analyze video and output game states."""
    extractor = GameStateExtractor()

    with VideoFrameExtractor(args.video) as video:
        start_ms = args.start * 1000
        end_ms = args.end * 1000 if args.end else None

        if args.verbose:
            print(f"Analyzing {args.video}", file=sys.stderr)
            print(f"  Duration: {video.duration_seconds:.1f}s", file=sys.stderr)

        results = []
        count = 0

        for frame_info in video.iterate_at_interval(args.interval, start_ms, end_ms):
            state = extractor.extract(
                frame_info.image,
                frame_number=frame_info.frame_number,
                timestamp_ms=frame_info.timestamp_ms
            )

            result = game_state_to_dict(state)
            results.append(result)
            count += 1

            if args.verbose and count % 10 == 0:
                print(f"  Analyzed {count} frames...", file=sys.stderr)

        if args.verbose:
            print(f"Done! Analyzed {count} frames.", file=sys.stderr)

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

        # Collect all game states
        states = []
        for frame_info in video.iterate_at_interval(args.interval):
            state = extractor.extract(
                frame_info.image,
                frame_number=frame_info.frame_number,
                timestamp_ms=frame_info.timestamp_ms
            )
            states.append(state)

        print(f"  Collected {len(states)} game states", file=sys.stderr)

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
