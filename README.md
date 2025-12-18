# Lieutenant of Poker

Video analysis software for CoinPoker screencaps. Extracts game state information from video frames using computer vision (OpenCV) and OCR (Tesseract), then exports hand histories in standard poker formats.

## Features

- Detects players, chip counts, blinds, and dealer button position
- Recognizes hero and community cards via template matching
- Tracks game state changes frame-by-frame
- Exports to Snowie and human-readable formats
- Splits long recordings into individual hands
- Batch processes entire folders of videos

## Installation

Requires Python 3.10+ and Tesseract OCR.

```bash
# Install Tesseract (macOS)
brew install tesseract

# Install the package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Auto mode (record + export)

```bash
# Record with auto-detection and automatically export completed hands
lieutenant auto --output-dir ./session
```

This runs screen capture with auto-detection (starts/stops recording based on hero card visibility) and automatically exports each completed video to Snowie format.

### Export a hand history

```bash
# Export to Snowie format
lieutenant export video.mp4 --format snowie

# Human-readable summary
lieutenant export video.mp4 --format human
```

### Batch export

```bash
# Export all videos in a folder
lieutenant batch-export ./videos --format snowie --output ./exports
```

### Split a long recording

```bash
# Split into individual hands based on hero card detection
lieutenant split recording.mp4 --output ./hands
```

### Analyze a video

```bash
# Show game state changes
lieutenant analyze video.mp4

# Verbose mode (show rejected frames)
lieutenant analyze video.mp4 --verbose
```

### Video info

```bash
# Show video metadata and detected table info
lieutenant info video.mp4
```

### Diagnostics

```bash
# Generate detailed HTML report for debugging
lieutenant diagnose video.mp4 --open

# Analyze a specific timestamp
lieutenant diagnose video.mp4 --timestamp 5.0
```

### Screen recording (macOS only)

```bash
# Record screen to file
lieutenant record output.mp4
```

## Export Formats

| Format | Description |
|--------|-------------|
| `snowie` | PokerSnowie import format |
| `human` | Human-readable summary |
| `actions` | Simple action log |

## Development

```bash
# Format code
black .

# Lint
ruff check .
pylint lieutenant_of_poker
```

## License

MIT
