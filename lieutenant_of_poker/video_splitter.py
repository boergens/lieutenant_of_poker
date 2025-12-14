"""
Video splitting based on hero card detection.

Detects segments where the hero's cards are visible and the hero name is
displayed, and splits the video into separate chunk files using ffmpeg.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# 100 memorable animal names for chunk identification
ANIMAL_NAMES = [
    "antelope", "badger", "beaver", "bison", "bobcat",
    "buffalo", "camel", "cardinal", "caribou", "cheetah",
    "chipmunk", "cobra", "condor", "cougar", "coyote",
    "crane", "crow", "deer", "dingo", "dolphin",
    "donkey", "eagle", "elephant", "elk", "falcon",
    "ferret", "finch", "flamingo", "fox", "gazelle",
    "gecko", "giraffe", "goat", "goose", "gorilla",
    "grizzly", "gull", "hamster", "hare", "hawk",
    "hedgehog", "heron", "hippo", "hornet", "horse",
    "hound", "hyena", "ibex", "iguana", "impala",
    "jackal", "jaguar", "jay", "kangaroo", "kestrel",
    "kingfish", "kite", "koala", "lemur", "leopard",
    "lion", "lizard", "llama", "lobster", "lynx",
    "macaw", "magpie", "marmot", "marten", "meerkat",
    "moose", "mouse", "narwhal", "newt", "ocelot",
    "octopus", "orca", "osprey", "otter", "owl",
    "panda", "panther", "parrot", "pelican", "penguin",
    "pheasant", "porcupine", "puma", "quail", "rabbit",
    "raccoon", "raven", "rhino", "robin", "salmon",
    "scorpion", "seal", "shark", "sheep", "shrimp",
]


@dataclass
class Segment:
    """A detected video segment."""
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000


@dataclass
class SplitResult:
    """Result of a split operation."""
    segments: list[Segment]
    created_files: list[Path]
    failed_files: list[tuple[Path, str]]  # (path, error_message)


def get_used_animals(output_dir: Path) -> set[str]:
    """Scan output directory for already-used animal names."""
    used = set()
    for mp4 in output_dir.glob("*.mp4"):
        # Parse filenames like: prefix_20251202_001_antelope.mp4
        parts = mp4.stem.split("_")
        if parts:
            # Last part before extension is the animal name
            animal = parts[-1].lower()
            if animal in ANIMAL_NAMES:
                used.add(animal)
    return used


def pick_animal(output_dir: Path) -> str:
    """Pick an unused animal name for this session."""
    used = get_used_animals(output_dir)
    for animal in ANIMAL_NAMES:
        if animal not in used:
            return animal
    # All used - start over with numeric suffix
    return f"{ANIMAL_NAMES[0]}2"


def split_video(
    video_path: Path,
    segments: list[Segment],
    output_dir: Path,
    prefix: str,
    on_chunk: Optional[Callable[[int, int, Path], None]] = None,
) -> SplitResult:
    """
    Split a video into chunks using ffmpeg.

    Filenames are: {prefix}_{date}_{num}_{animal}.mp4
    e.g., gop3_20251202_001_antelope.mp4

    Args:
        video_path: Path to the source video.
        segments: List of segments to extract.
        output_dir: Directory for output files.
        prefix: Filename prefix for chunks.
        on_chunk: Optional callback(chunk_num, total_chunks, output_path).

    Returns:
        SplitResult with created files and any failures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get date
    date_str = datetime.now().strftime("%Y%m%d")

    created_files = []
    failed_files = []

    for i, segment in enumerate(segments):
        chunk_num = i + 1
        # Pick a unique animal name for each chunk
        animal = pick_animal(output_dir)
        # Format: prefix_date_num_animal.mp4
        output_file = output_dir / f"{prefix}_{date_str}_{chunk_num:03d}_{animal}.mp4"

        if on_chunk:
            on_chunk(chunk_num, len(segments), output_file)

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(segment.start_ms / 1000),  # Start time in seconds
            "-i", str(video_path),
            "-t", str(segment.duration_s),  # Duration
            "-c:v", "libx264",  # Re-encode with H.264
            "-preset", "fast",  # Encoding speed
            "-crf", "18",  # Quality (lower = better, 18 is visually lossless)
            "-c:a", "copy",  # Copy audio stream
            "-avoid_negative_ts", "make_zero",
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            created_files.append(output_file)
        else:
            error = result.stderr[:200] if result.stderr else "Unknown error"
            failed_files.append((output_file, error))

    return SplitResult(
        segments=segments,
        created_files=created_files,
        failed_files=failed_files,
    )


def split_video_by_hero(
    video_path: str,
    output_dir: Optional[str],
    prefix: Optional[str],
    dry_run: bool,
) -> None:
    """Split video into chunks based on hero card detection (CLI entry point)."""
    import sys
    from .frame_extractor import VideoFrameExtractor
    from .progress import create_progress
    from ._detection import detect_hero_cards

    video_path = Path(video_path)
    out_dir = Path(output_dir) if output_dir else video_path.parent
    file_prefix = prefix if prefix else video_path.stem

    print(f"Analyzing {video_path}...", file=sys.stderr)

    # Detect segments by scanning for hero cards
    segments = []
    in_segment = False
    segment_start_ms = 0.0
    consecutive_needed = 3
    history = []

    with create_progress() as progress:
        with VideoFrameExtractor(str(video_path)) as extractor:
            total_frames = extractor.frame_count
            task = progress.add_task("Scanning for hero", total=total_frames)

            for i, frame_info in enumerate(extractor.iterate_frames()):
                progress.update(task, completed=i)

                hero_found = detect_hero_cards(frame_info.image)
                history.append((hero_found, frame_info.timestamp_ms))
                if len(history) > consecutive_needed:
                    history.pop(0)

                if len(history) >= consecutive_needed:
                    all_found = all(h[0] for h in history)
                    none_found = not any(h[0] for h in history)

                    if all_found and not in_segment:
                        in_segment = True
                        segment_start_ms = history[0][1]
                    elif none_found and in_segment:
                        in_segment = False
                        segment_end_ms = history[0][1]
                        segments.append(Segment(segment_start_ms, segment_end_ms))

            # Close any open segment
            if in_segment:
                segments.append(Segment(segment_start_ms, frame_info.timestamp_ms))

            progress.update(task, completed=total_frames)

    print(f"\nDetected {len(segments)} segments:", file=sys.stderr)
    for i, seg in enumerate(segments):
        print(f"  {i+1}. {seg.start_ms/1000:.2f}s - {seg.end_ms/1000:.2f}s ({seg.duration_s:.1f}s)", file=sys.stderr)

    if not segments:
        print("\nNo segments found.", file=sys.stderr)
        return

    if dry_run:
        print("\nDry run - no files created.", file=sys.stderr)
        return

    print(f"\nSplitting into {len(segments)} chunks...", file=sys.stderr)

    def on_chunk(num, total, path):
        print(f"  Creating {path.name}...", file=sys.stderr, end=" ", flush=True)

    result = split_video(video_path, segments, out_dir, file_prefix, on_chunk)

    for _ in result.created_files:
        print("OK", file=sys.stderr)
    for _, error in result.failed_files:
        print(f"FAILED: {error}", file=sys.stderr)

    print(f"\nCreated {len(result.created_files)} chunk(s) in {out_dir}/", file=sys.stderr)
