"""
Video splitting based on brightness detection.

Uses the BrightnessDetector to find segments where the screen is "on"
and splits the video into separate chunk files using ffmpeg.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .frame_extractor import VideoFrameExtractor
from .video_recorder import BrightnessDetector
from .table_regions import BASE_WIDTH, BASE_HEIGHT

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


def _get_video_info(video_path: Path) -> tuple[int, int, float, float]:
    """Get video width, height, fps, duration using ffprobe."""
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    # Parse frame rate like "60/1" or "29.97"
    fps_str = stream["r_frame_rate"]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    duration = float(stream.get("duration", 0))
    return width, height, fps, duration


def _extract_frames_at_fps(video_path: Path, target_fps: float = 1.0) -> list[tuple[float, bool]]:
    """
    Extract frames at target_fps and classify as bright/dark.
    Returns list of (timestamp_seconds, is_bright).
    """
    import numpy as np

    width, height, _, duration = _get_video_info(video_path)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={target_fps}",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-loglevel", "error",
        "pipe:1"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = width * height
    threshold = 250.0

    results = []
    timestamp = 0.0
    interval = 1.0 / target_fps

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        # Check center region
        h, w = frame.shape
        center = frame[h//3:2*h//3, w//3:2*w//3]
        is_bright = float(np.mean(center)) > threshold

        results.append((timestamp, is_bright))
        timestamp += interval

    proc.wait()
    return results


def _refine_transition_ffmpeg(
    video_path: Path,
    lo_sec: float,
    hi_sec: float,
    looking_for_bright: bool,
    consecutive_frames: int,
    fps: float,
) -> float:
    """
    Binary search using ffmpeg to find exact transition point.
    Returns timestamp in seconds.
    """
    import numpy as np

    width, height, _, _ = _get_video_info(video_path)
    threshold = 250.0
    frame_duration = 1.0 / fps

    def is_bright_at(t: float) -> bool:
        cmd = [
            "ffmpeg", "-ss", str(t), "-i", str(video_path),
            "-vframes", "1",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-loglevel", "error",
            "pipe:1"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if len(result.stdout) < width * height:
            return False
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width))
        h, w = frame.shape
        center = frame[h//3:2*h//3, w//3:2*w//3]
        return float(np.mean(center)) > threshold

    # Binary search
    while hi_sec - lo_sec > frame_duration * 2:
        mid = (lo_sec + hi_sec) / 2
        bright = is_bright_at(mid)
        if bright == looking_for_bright:
            hi_sec = mid
        else:
            lo_sec = mid

    # Verify consecutive frames rule
    candidate = hi_sec
    for offset in range(-consecutive_frames, consecutive_frames + 1):
        t = candidate + offset * frame_duration
        if t < 0:
            continue
        all_match = True
        for i in range(consecutive_frames):
            if is_bright_at(t + i * frame_duration) != looking_for_bright:
                all_match = False
                break
        if all_match:
            return t

    return candidate


def detect_segments(
    video_path: Path,
    threshold: float = 250.0,
    consecutive_frames: int = 3,
    step: int = 1,  # kept for API compatibility, ignored
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[Segment]:
    """
    Detect bright segments in a video using coarse-to-fine search.

    Algorithm:
    1. Use ffmpeg to extract frames at 1fps for coarse classification
    2. Find connected components of bright regions
    3. Binary search with ffmpeg to find exact transition points

    Args:
        video_path: Path to the video file.
        threshold: Brightness threshold 0-255 (default: 250).
        consecutive_frames: Frames needed to trigger state change (default: 3).
        step: Ignored (kept for API compatibility).
        on_progress: Optional callback(current_sample, total_samples).

    Returns:
        List of detected Segment objects.
    """
    _, _, fps, duration = _get_video_info(video_path)

    if on_progress:
        on_progress(0, int(duration))

    # Step 1: Coarse sampling at 1fps using ffmpeg (fast!)
    samples = _extract_frames_at_fps(video_path, target_fps=1.0)

    if on_progress:
        on_progress(len(samples), len(samples))

    # Step 2: Find connected components (runs of bright samples)
    candidate_regions = []  # List of (start_sec, end_sec)
    in_region = False
    region_start = 0.0

    for timestamp, is_bright in samples:
        if is_bright and not in_region:
            in_region = True
            region_start = timestamp
        elif not is_bright and in_region:
            in_region = False
            candidate_regions.append((region_start, timestamp))

    # Handle region that extends to end
    if in_region:
        candidate_regions.append((region_start, duration))

    # Step 3: Refine boundaries with binary search
    segments = []
    for start_sec, end_sec in candidate_regions:
        # Find exact start
        if start_sec <= 0:
            exact_start = 0.0
        else:
            exact_start = _refine_transition_ffmpeg(
                video_path, start_sec - 1.0, start_sec,
                looking_for_bright=True, consecutive_frames=consecutive_frames, fps=fps
            )

        # Find exact end
        if end_sec >= duration:
            exact_end = duration
        else:
            exact_end = _refine_transition_ffmpeg(
                video_path, end_sec - 1.0, end_sec,
                looking_for_bright=False, consecutive_frames=consecutive_frames, fps=fps
            )

        segments.append(Segment(exact_start * 1000, exact_end * 1000))

    return segments


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
    min_duration_s: float = 1.0,
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
        min_duration_s: Minimum segment duration in seconds (default: 1.0).
        on_chunk: Optional callback(chunk_num, total_chunks, output_path).

    Returns:
        SplitResult with created files and any failures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by minimum duration
    min_duration_ms = min_duration_s * 1000
    filtered = [s for s in segments if s.duration_ms >= min_duration_ms]

    # Pick animal name for this session and get date
    animal = pick_animal(output_dir)
    date_str = datetime.now().strftime("%Y%m%d")

    created_files = []
    failed_files = []

    for i, segment in enumerate(filtered):
        chunk_num = i + 1
        # Format: prefix_date_num_animal.mp4
        output_file = output_dir / f"{prefix}_{date_str}_{chunk_num:03d}_{animal}.mp4"

        if on_chunk:
            on_chunk(chunk_num, len(filtered), output_file)

        # Scale to target resolution (BASE_WIDTH x BASE_HEIGHT)
        # Using scale filter that maintains aspect ratio and pads if needed
        scale_filter = (
            f"scale={BASE_WIDTH}:{BASE_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={BASE_WIDTH}:{BASE_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        )

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(segment.start_ms / 1000),  # Start time in seconds
            "-i", str(video_path),
            "-t", str(segment.duration_s),  # Duration
            "-vf", scale_filter,  # Scale to target resolution
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
        segments=filtered,
        created_files=created_files,
        failed_files=failed_files,
    )


def split_video_by_brightness(
    video_path: str,
    output_dir: Optional[str],
    prefix: Optional[str],
    threshold: float,
    consecutive: int,
    min_duration: float,
    step: int,
    dry_run: bool,
) -> None:
    """Split video into chunks based on brightness detection (CLI entry point)."""
    import sys
    from .frame_extractor import get_video_info
    from .progress import create_progress

    video_path = Path(video_path)
    out_dir = Path(output_dir) if output_dir else video_path.parent
    file_prefix = prefix if prefix else video_path.stem

    info = get_video_info(str(video_path))
    total_frames = info['frame_count']

    print(f"Analyzing {video_path}...", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)
    print(f"  Threshold: {threshold}, Consecutive: {consecutive}", file=sys.stderr)

    with create_progress() as progress:
        task = progress.add_task("Scanning frames", total=total_frames)
        segments = detect_segments(
            video_path,
            threshold=threshold,
            consecutive_frames=consecutive,
            step=step,
            on_progress=lambda c, t: progress.update(task, completed=c),
        )

    min_ms = min_duration * 1000
    filtered = [s for s in segments if s.duration_ms >= min_ms]

    print(f"\nDetected {len(segments)} segments ({len(filtered)} >= {min_duration}s):", file=sys.stderr)
    for i, seg in enumerate(filtered):
        print(f"  {i+1}. {seg.start_ms/1000:.2f}s - {seg.end_ms/1000:.2f}s ({seg.duration_s:.1f}s)", file=sys.stderr)

    if not filtered:
        print("\nNo segments found.", file=sys.stderr)
        return

    if dry_run:
        print("\nDry run - no files created.", file=sys.stderr)
        return

    print(f"\nSplitting into {len(filtered)} chunks...", file=sys.stderr)

    def on_chunk(num, total, path):
        print(f"  Creating {path.name}...", file=sys.stderr, end=" ", flush=True)

    result = split_video(video_path, segments, out_dir, file_prefix, min_duration, on_chunk)

    for _ in result.created_files:
        print("OK", file=sys.stderr)
    for _, error in result.failed_files:
        print(f"FAILED: {error}", file=sys.stderr)

    print(f"\nCreated {len(result.created_files)} chunk(s) in {out_dir}/", file=sys.stderr)
