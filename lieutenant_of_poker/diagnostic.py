"""
Diagnostic report generator for frame analysis.

Creates detailed HTML reports showing each step of the detection pipeline
with images, intermediate results, and OCR outputs.
"""

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class DiagnosticStep:
    """A single step in the diagnostic process."""
    name: str
    description: str
    images: List[tuple[str, np.ndarray]] = field(default_factory=list)
    ocr_result: Optional[str] = None
    match_info: Optional[str] = None
    parsed_result: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    substeps: List["DiagnosticStep"] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a frame."""
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: Optional[int] = None
    frame_timestamp_ms: Optional[float] = None
    frame_size: tuple[int, int] = (0, 0)
    frame: Optional[np.ndarray] = None
    steps: List[DiagnosticStep] = field(default_factory=list)


def _image_to_base64(image: np.ndarray, max_width: int = 400) -> str:
    """Convert numpy image to base64 for HTML embedding."""
    if image is None or image.size == 0:
        return ""

    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class DiagnosticExtractor:
    """
    Game state extractor that captures diagnostic information.
    """

    def __init__(self, hero_position: Optional[tuple[int, int]] = None):
        """
        Initialize the diagnostic extractor.

        Args:
            hero_position: (x, y) of hero's player position for card detection.
        """
        self.hero_position = hero_position
        self.report: Optional[DiagnosticReport] = None

    def extract_with_diagnostics(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ) -> DiagnosticReport:
        """
        Extract game state with full diagnostic capture.
        """
        self.report = DiagnosticReport(
            frame_number=frame_number,
            frame_timestamp_ms=timestamp_ms,
            frame_size=(frame.shape[1], frame.shape[0]),
            frame=frame.copy(),
        )

        self._extract_community_cards(frame)
        if self.hero_position:
            self._extract_hero_cards(frame)

        return self.report

    def _extract_community_cards(self, frame: np.ndarray) -> None:
        """Extract community cards with diagnostics."""
        step = DiagnosticStep(
            name="Community Cards Detection",
            description="Detecting cards on the board",
        )

        try:
            from .card_matcher import (
                COMMUNITY_LIBRARY, _match_rank, _match_suit,
                extract_community_card_regions, match_community_cards,
            )

            regions = extract_community_card_regions(frame)
            matched_cards = match_community_cards(frame)

            cards = []
            for i, (rank_img, suit_img) in enumerate(regions.cards):
                substep = DiagnosticStep(
                    name=f"Slot {i+1}",
                    description=f"Card slot {i+1} of 5",
                )
                substep.images.append(("Rank Region", rank_img))
                substep.images.append(("Suit Region", suit_img))

                rank = _match_rank(rank_img, COMMUNITY_LIBRARY)
                suit = _match_suit(suit_img, COMMUNITY_LIBRARY)

                substep.match_info = f"Rank: {rank or 'not found'}, Suit: {suit or 'not found'}"

                card = matched_cards[i]
                if card:
                    substep.parsed_result = card
                    substep.success = True
                    cards.append(card)
                else:
                    substep.parsed_result = "(not detected)"
                    substep.success = True

                step.substeps.append(substep)

            step.parsed_result = cards if cards else []
            step.success = True
            step.description += f" - Found {len(cards)} cards"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_hero_cards(self, frame: np.ndarray) -> None:
        """Extract hero cards with diagnostics."""
        step = DiagnosticStep(
            name="Hero Cards Detection",
            description=f"Detecting hero's hole cards at position {self.hero_position}",
        )

        try:
            from .card_matcher import (
                HERO_LIBRARY, _match_rank, _match_suit,
                HERO_RANK_OFFSET, HERO_SUIT_OFFSET, HERO_RANK_SIZE, HERO_SUIT_SIZE,
                HERO_CARD_SPACING, match_hero_cards,
            )

            px, py = self.hero_position

            # Calculate regions for display
            left_rank_region = (px + HERO_RANK_OFFSET[0], py + HERO_RANK_OFFSET[1], *HERO_RANK_SIZE)
            left_suit_region = (px + HERO_SUIT_OFFSET[0], py + HERO_SUIT_OFFSET[1], *HERO_SUIT_SIZE)

            def extract(r):
                x, y, w, h = r
                return frame[y:y+h, x:x+w]

            left_rank_img = extract(left_rank_region)
            left_suit_img = extract(left_suit_region)
            right_rank_img = extract((left_rank_region[0] + HERO_CARD_SPACING, *left_rank_region[1:]))
            right_suit_img = extract((left_suit_region[0] + HERO_CARD_SPACING, *left_suit_region[1:]))

            matched_cards = match_hero_cards(frame, self.hero_position)
            cards = []

            # LEFT CARD
            left_substep = DiagnosticStep(name="Left Card", description="Hero left card")
            left_substep.images.append(("Rank Region", left_rank_img))
            left_substep.images.append(("Suit Region", left_suit_img))

            left_rank = _match_rank(left_rank_img, HERO_LIBRARY)
            left_suit = _match_suit(left_suit_img, HERO_LIBRARY)
            left_substep.match_info = f"Rank: {left_rank or 'not found'}, Suit: {left_suit or 'not found'}"

            left_card = matched_cards[0]
            if left_card:
                left_substep.parsed_result = left_card
                left_substep.success = True
                cards.append(left_card)
            else:
                left_substep.parsed_result = "(not detected)"
                left_substep.success = False

            step.substeps.append(left_substep)

            # RIGHT CARD
            right_substep = DiagnosticStep(name="Right Card", description="Hero right card")
            right_substep.images.append(("Rank Region", right_rank_img))
            right_substep.images.append(("Suit Region", right_suit_img))

            right_rank = _match_rank(right_rank_img, HERO_LIBRARY)
            right_suit = _match_suit(right_suit_img, HERO_LIBRARY)
            right_substep.match_info = f"Rank: {right_rank or 'not found'}, Suit: {right_suit or 'not found'}"

            right_card = matched_cards[1]
            if right_card:
                right_substep.parsed_result = right_card
                right_substep.success = True
                cards.append(right_card)
            else:
                right_substep.parsed_result = "(not detected)"
                right_substep.success = False

            step.substeps.append(right_substep)

            step.parsed_result = cards if cards else []
            step.success = all(s.success for s in step.substeps)
            step.description += f" - Found {len(cards)} cards"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)


def generate_html_report(report: DiagnosticReport, output_path: Optional[Path] = None) -> str:
    """Generate an HTML report from diagnostic data."""
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Frame Analysis Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .step {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .step.success { border-left: 4px solid #4CAF50; }
        .step.failure { border-left: 4px solid #f44336; }
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .step-name { font-weight: bold; font-size: 1.1em; }
        .badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
        }
        .badge.success { background: #e8f5e9; color: #2e7d32; }
        .badge.failure { background: #ffebee; color: #c62828; }
        .description { color: #666; margin: 10px 0; }
        .images {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 15px 0;
        }
        .image-container { text-align: center; }
        .image-container img {
            max-width: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-label {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        .ocr-result {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            margin: 10px 0;
        }
        .parsed-result { font-weight: bold; color: #1976d2; }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .substeps {
            margin-left: 30px;
            border-left: 2px solid #e0e0e0;
            padding-left: 20px;
        }
        .meta {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .meta-item { color: #666; }
        .meta-value { font-weight: bold; color: #333; }
        .frames-preview img {
            max-width: 600px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
"""]

    html_parts.append("<h1>🎰 Frame Analysis Report</h1>")

    # Metadata
    html_parts.append('<div class="meta"><div class="meta-grid">')
    html_parts.append(f'<div class="meta-item">Generated: <span class="meta-value">{report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span></div>')
    if report.frame_number is not None:
        html_parts.append(f'<div class="meta-item">Frame #: <span class="meta-value">{report.frame_number}</span></div>')
    if report.frame_timestamp_ms is not None:
        html_parts.append(f'<div class="meta-item">Timestamp: <span class="meta-value">{report.frame_timestamp_ms/1000:.2f}s</span></div>')
    html_parts.append(f'<div class="meta-item">Frame Size: <span class="meta-value">{report.frame_size[0]}x{report.frame_size[1]}</span></div>')
    html_parts.append('</div></div>')

    # Frame preview
    html_parts.append('<h2>Frame Preview</h2>')
    if report.frame is not None:
        b64 = _image_to_base64(report.frame, max_width=700)
        html_parts.append(f'<div class="image-container"><img src="data:image/png;base64,{b64}"></div>')

    # Steps
    html_parts.append('<h2>Detection Steps</h2>')
    for step in report.steps:
        html_parts.append(_render_step(step))

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)

    if output_path:
        Path(output_path).write_text(html)

    return html


def _render_step(step: DiagnosticStep) -> str:
    """Render a single step as HTML."""
    status_class = "success" if step.success else "failure"
    status_text = "✓ Success" if step.success else "✗ Failed"

    html = f'<div class="step {status_class}">'
    html += f'<div class="step-header"><span class="step-name">{step.name}</span>'
    html += f'<span class="badge {status_class}">{status_text}</span></div>'
    html += f'<div class="description">{step.description}</div>'

    if step.images:
        html += '<div class="images">'
        for label, img in step.images:
            if img is not None and img.size > 0:
                b64 = _image_to_base64(img)
                html += f'<div class="image-container"><img src="data:image/png;base64,{b64}">'
                html += f'<div class="image-label">{label}</div></div>'
        html += '</div>'

    if step.ocr_result is not None:
        html += f'<div class="ocr-result">OCR Result: <code>{step.ocr_result}</code></div>'

    if step.match_info is not None:
        html += f'<div class="ocr-result">Match: <code>{step.match_info}</code></div>'

    if step.parsed_result is not None:
        html += f'<div class="parsed-result">Parsed: {step.parsed_result}</div>'

    if step.error:
        html += f'<div class="error">Error: {step.error}</div>'

    if step.substeps:
        html += '<div class="substeps">'
        for substep in step.substeps:
            html += _render_step(substep)
        html += '</div>'

    html += '</div>'
    return html


def generate_diagnostic_report(
    video_path: str,
    output_path: Path,
    frame_number: Optional[int] = None,
    timestamp_s: Optional[float] = None,
) -> dict:
    """
    Generate a diagnostic report for a specific frame.

    Args:
        video_path: Path to the video file.
        output_path: Path for the HTML report.
        frame_number: Frame number to analyze (takes precedence over timestamp_s).
        timestamp_s: Timestamp in seconds.

    Returns:
        Dictionary with report statistics.
    """
    from .first_frame import TableInfo
    from .frame_extractor import VideoFrameExtractor

    # Get hero position from TableInfo
    table_info = TableInfo.from_video(video_path)
    hero_position = table_info.positions[-1] if table_info.positions else None

    with VideoFrameExtractor(video_path) as video:
        if frame_number is not None:
            frame_info = video.get_frame_at(frame_number)
        else:
            frame_info = video.get_frame_at_timestamp((timestamp_s or 0) * 1000)

        if frame_info is None:
            raise ValueError("Could not read frame")

        extractor = DiagnosticExtractor(hero_position=hero_position)
        report = extractor.extract_with_diagnostics(
            frame_info.image,
            frame_number=frame_info.frame_number,
            timestamp_ms=frame_info.timestamp_ms,
        )

        generate_html_report(report, output_path)

        successes = sum(1 for s in report.steps if s.success)
        failures = sum(1 for s in report.steps if not s.success)

        return {
            "frame_number": frame_info.frame_number,
            "timestamp_ms": frame_info.timestamp_ms,
            "hero_position": hero_position,
            "steps_succeeded": successes,
            "steps_failed": failures,
            "output_path": output_path,
        }


def diagnose(
    video_path: str,
    output: str = "diagnostic_report.html",
    frame_number: Optional[int] = None,
    timestamp_s: Optional[float] = None,
    open_browser: bool = False,
) -> None:
    """Generate diagnostic report and optionally open in browser."""
    import sys
    import webbrowser
    from .frame_extractor import get_video_info

    info = get_video_info(video_path)
    print(f"Video: {video_path}", file=sys.stderr)
    print(f"  Resolution: {info['width']}x{info['height']}", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)

    output_path = Path(output)
    result = generate_diagnostic_report(
        video_path,
        output_path,
        frame_number=frame_number,
        timestamp_s=timestamp_s,
    )

    print(f"\nAnalyzing frame {result['frame_number']} ({result['timestamp_ms']/1000:.2f}s)...", file=sys.stderr)
    print(f"Report generated: {output_path}", file=sys.stderr)
    print(f"  Steps: {result['steps_succeeded']} succeeded, {result['steps_failed']} failed", file=sys.stderr)

    if open_browser:
        webbrowser.open(f"file://{output_path.absolute()}")
