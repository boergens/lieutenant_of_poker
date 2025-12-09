"""Debug helper to step through export logic in IDE."""

from lieutenant_of_poker.analysis import analyze_video
from lieutenant_of_poker.frame_extractor import get_video_info
from lieutenant_of_poker.first_frame import TableInfo
from lieutenant_of_poker.action_log_export import export_action_log
from lieutenant_of_poker.export import reconstruct_hand


def debug_export(video_path: str, format: str = "actions"):
    """Run export pipeline - set breakpoints here to debug."""

    # Get video info
    info = get_video_info(video_path)
    print(f"Video: {video_path}")
    print(f"  Duration: {info['duration_seconds']:.1f}s, FPS: {info['fps']}")

    # Detect button and players from first frame
    table = TableInfo.from_video(video_path)
    button_pos = table.button_index if table.button_index is not None else 0
    players = list(table.names)
    print(f"  Button: {button_pos}")
    print(f"  Players: {players}")

    # Analyze video
    states = analyze_video(video_path)
    print(f"  States: {len(states)}")

    # Find hero cards from states
    hero_cards = []
    for state in states:
        if state.get("hero_cards"):
            hero_cards = state["hero_cards"]
            break

    # Reconstruct hand (this is where actions are built)
    hand = reconstruct_hand(states, players, button_pos, hero_cards)

    # Export
    output = export_action_log(states, button_pos=button_pos, player_names=players)
    print("\n" + output)

    return hand, states


if __name__ == "__main__":
    hand, states = debug_export("testvid.mp4")
