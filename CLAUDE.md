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

# Format code
black .

# Lint
ruff check .
ruff check --fix .
pylint lieutenant_of_poker
```

## External Dependencies

Tesseract OCR must be installed on the system:
```bash
brew install tesseract  # macOS
```

## Notes

- It is possible to break Claude by looking at large images - please make sure to always check that an image is below a Megabyte first before looking at it
