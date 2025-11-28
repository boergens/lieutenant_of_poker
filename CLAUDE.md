# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lieutenant of Poker is a Python library for analyzing screencap videos of Governor of Poker. It uses computer vision (OpenCV) and OCR (pytesseract) to extract game state information from video frames.

## Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install package in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_file.py::test_function_name

# Format code
black .

# Lint
ruff check .
ruff check --fix .
```

## External Dependencies

Tesseract OCR must be installed on the system:
```bash
brew install tesseract  # macOS
```

## Task Tracking

This project uses **beads** for issue tracking instead of markdown TODOs or external tools. Key commands:
- `bd ready` - Find available work
- `bd create --title="..." --type=task|bug|feature` - Create new issue
- `bd update <id> --status=in_progress` - Claim work
- `bd close <id>` - Complete work
- `bd sync --from-main` - Sync beads from main branch
- it is possible to break Claude by looking at large images.. please make sure to always check that an image is below a Megabyte first before looking at it
- make a git commit whenever you close a bead issue