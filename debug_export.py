"""Debug helper to step through export logic in IDE."""

from lieutenant_of_poker.analysis import analyze_video
from lieutenant_of_poker.frame_extractor import get_video_info
from lieutenant_of_poker.first_frame import TableInfo
from lieutenant_of_poker.action_log_export import format_action_log
from lieutenant_of_poker.export import reconstruct_hand


def debug_export(video_path: str):
    """Run export pipeline - set breakpoints here to debug."""

    # Get video info
    info = get_video_info(video_path)
    print(f"Video: {video_path}")
    print(f"  Duration: {info['duration_seconds']:.1f}s, FPS: {info['fps']}")

    # Detect button and players from first frame
    table = TableInfo.from_video(video_path)
    print(f"  Button: {table.button_index}")
    print(f"  Players: {list(table.names)}")
    print(f"  Hero cards: {list(table.hero_cards)}")

    # Analyze video
    states = analyze_video(video_path)
    print(f"  States: {len(states)}")

    # Reconstruct hand (this is where actions are built)
    hand = reconstruct_hand(states, table)

    # Export
    output = format_action_log(hand)
    print("\n" + output)

    return hand, states


if __name__ == "__main__":
    hand, states = debug_export("testvid.mp4")
