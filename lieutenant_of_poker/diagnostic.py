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

from lieutenant_of_poker.table_regions import BASE_WIDTH, BASE_HEIGHT


@dataclass
class DiagnosticStep:
    """A single step in the diagnostic process."""
    name: str
    description: str
    images: List[tuple[str, np.ndarray]] = field(default_factory=list)  # (label, image)
    ocr_result: Optional[str] = None  # For OCR-based detection (pot, chips)
    match_info: Optional[str] = None  # For library-based card matching
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
    original_size: tuple[int, int] = (0, 0)
    scaled_size: tuple[int, int] = (0, 0)
    original_frame: Optional[np.ndarray] = None
    scaled_frame: Optional[np.ndarray] = None
    steps: List[DiagnosticStep] = field(default_factory=list)
    final_state: Optional[dict] = None


def _image_to_base64(image: np.ndarray, max_width: int = 400) -> str:
    """Convert numpy image to base64 for HTML embedding."""
    if image is None or image.size == 0:
        return ""

    # Scale down if too large
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert BGR to RGB if color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL and then to base64
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class DiagnosticExtractor:
    """
    Game state extractor that captures diagnostic information.

    Wraps the normal extraction process and records each step
    for later report generation.
    """

    def __init__(self):
        """Initialize the diagnostic extractor."""
        from .table_regions import detect_table_regions, PlayerPosition

        self._detect_regions = detect_table_regions
        self._PlayerPosition = PlayerPosition
        self.report: Optional[DiagnosticReport] = None

    def extract_with_diagnostics(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ) -> DiagnosticReport:
        """
        Extract game state with full diagnostic capture.

        Args:
            frame: BGR image frame.
            frame_number: Optional frame number.
            timestamp_ms: Optional timestamp.

        Returns:
            DiagnosticReport with all steps documented.
        """
        self.report = DiagnosticReport(
            frame_number=frame_number,
            frame_timestamp_ms=timestamp_ms,
            original_size=(frame.shape[1], frame.shape[0]),
            original_frame=frame.copy(),
        )

        # Scale frame if needed
        scaled_frame = self._scale_frame(frame)
        self.report.scaled_size = (scaled_frame.shape[1], scaled_frame.shape[0])
        self.report.scaled_frame = scaled_frame.copy()

        # Create region detector
        region_detector = self._detect_regions(scaled_frame)

        # Extract each component with diagnostics
        self._extract_pot(scaled_frame, region_detector)
        self._extract_community_cards(scaled_frame, region_detector)
        self._extract_hero_cards(scaled_frame, region_detector)
        self._extract_players(scaled_frame, region_detector)
        self._extract_player_names(scaled_frame)

        return self.report

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Scale frame and record the step."""
        h, w = frame.shape[:2]
        target_w, target_h = BASE_WIDTH, BASE_HEIGHT

        step = DiagnosticStep(
            name="Frame Scaling",
            description=f"Original size: {w}x{h}",
        )
        step.images.append(("Original Frame", frame))

        if w <= target_w and h <= target_h:
            step.description += f" (no scaling needed)"
            step.parsed_result = "No scaling"
            self.report.steps.append(step)
            return frame

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        step.description += f" → Scaled to: {new_w}x{new_h} (scale={scale:.2f})"
        step.images.append(("Scaled Frame", scaled))
        step.parsed_result = f"{new_w}x{new_h}"
        self.report.steps.append(step)

        return scaled

    def _extract_pot(self, frame: np.ndarray, region_detector) -> None:
        """Extract pot with diagnostics."""
        from .fast_ocr import preprocess_for_ocr
        from .chip_ocr import extract_pot

        step = DiagnosticStep(
            name="Pot Detection",
            description="Extracting pot amount from center of table",
        )

        pot_region = region_detector.extract_pot(frame)
        step.images.append(("Pot Region", pot_region))
        step.images.append(("OCR Input (inverted)", preprocess_for_ocr(pot_region)))

        amount = extract_pot(frame, region_detector)
        step.parsed_result = amount
        step.success = amount is not None

        self.report.steps.append(step)

    def _extract_community_cards(self, frame: np.ndarray, region_detector) -> None:
        """Extract community cards with diagnostics using fixed slots and library matching."""
        step = DiagnosticStep(
            name="Community Cards Detection",
            description="Detecting cards on the board using 5 fixed slots + library matching",
        )

        try:
            # Show the overall region
            comm_region = region_detector.extract_community_cards(frame)
            step.images.append(("Community Cards Region", comm_region))

            # Extract each slot
            card_slots = region_detector.extract_community_card_slots(frame)
            from .card_detector import CardDetector
            from .card_matcher import get_card_matcher

            detector = CardDetector(use_library=False)  # For empty slot check only
            matcher = get_card_matcher()

            cards = []
            for i, slot_img in enumerate(card_slots):
                substep = DiagnosticStep(
                    name=f"Slot {i+1}",
                    description=f"Card slot {i+1} of 5",
                )
                substep.images.append((f"Slot {i+1}", slot_img))

                # First check if slot is empty
                if detector.is_empty_slot(slot_img):
                    substep.match_info = "Empty slot (matches table background)"
                    substep.parsed_result = "(empty slot)"
                    substep.success = True  # Empty is a valid state
                    step.substeps.append(substep)
                    continue

                # Extract and show rank/suit regions
                rank_region = matcher.extract_rank_region(slot_img)
                suit_region = matcher.extract_suit_region(slot_img)
                substep.images.append(("Rank Region", rank_region))
                substep.images.append(("Suit Region", suit_region))

                # Match rank
                rank = matcher.rank_matcher.match(rank_region)
                rank_info = f"Rank: {rank.value if rank else 'not found'}"

                # Match suit
                suit = matcher.suit_matcher.match(suit_region)
                suit_info = f"Suit: {suit.value if suit else 'not found'}"

                substep.match_info = f"{rank_info}, {suit_info}"

                if rank and suit:
                    from .card_detector import Card
                    card = Card(rank=rank, suit=suit)
                    substep.parsed_result = str(card)
                    substep.success = True
                    cards.append(card)
                else:
                    substep.parsed_result = "(incomplete detection)"
                    substep.success = False

                step.substeps.append(substep)

            step.parsed_result = [str(c) for c in cards] if cards else []
            step.success = True  # Success even with 0 cards (preflop)
            step.description += f" - Found {len(cards)} cards"

            # Show library stats
            stats = matcher.get_library_stats()
            step.description += f" (ranks: {stats['ranks']}/13, suits: {stats['suits']}/4)"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_hero_cards(self, frame: np.ndarray, region_detector) -> None:
        """Extract hero cards with diagnostics using calibrated subregions."""
        step = DiagnosticStep(
            name="Hero Cards Detection",
            description="Detecting hero's hole cards using calibrated subregions",
        )

        try:
            from .card_detector import Card, CardDetector
            from .card_matcher import (
                get_card_matcher, HERO_LEFT_LIBRARY, HERO_RIGHT_LIBRARY,
                HERO_LEFT_RANK_REGION, HERO_LEFT_SUIT_REGION,
                HERO_RIGHT_RANK_REGION, HERO_RIGHT_SUIT_REGION,
                _extract_region, _scale_hero_region,
            )

            # Show the overall region
            hero_region = region_detector.extract_hero_cards(frame)
            step.images.append(("Hero Cards Region", hero_region))

            # Get actual hero region size for scaling
            h, w = hero_region.shape[:2]
            hero_size = (w, h)

            # Scale regions to match actual hero region size
            left_rank_region = _scale_hero_region(HERO_LEFT_RANK_REGION, hero_size)
            left_suit_region = _scale_hero_region(HERO_LEFT_SUIT_REGION, hero_size)
            right_rank_region = _scale_hero_region(HERO_RIGHT_RANK_REGION, hero_size)
            right_suit_region = _scale_hero_region(HERO_RIGHT_SUIT_REGION, hero_size)

            # Background detector for empty slot check
            bg_detector = CardDetector(use_library=False)

            cards = []

            # LEFT CARD
            left_substep = DiagnosticStep(
                name="Left Card (hero_left)",
                description="Hero left card using hero_left library",
            )
            left_matcher = get_card_matcher(HERO_LEFT_LIBRARY)

            left_rank_img = _extract_region(hero_region, left_rank_region)
            left_suit_img = _extract_region(hero_region, left_suit_region)
            left_substep.images.append(("Left Rank Region", left_rank_img))
            left_substep.images.append(("Left Suit Region", left_suit_img))

            # Check if slot is empty (matches table background)
            if bg_detector.is_empty_slot(left_rank_img):
                left_substep.match_info = "Empty slot (matches table background)"
                left_substep.parsed_result = "(empty slot)"
                left_substep.success = True
            else:
                left_rank = left_matcher.rank_matcher.match(left_rank_img)
                left_suit = left_matcher.suit_matcher.match(left_suit_img)

                left_rank_info = f"Rank: {left_rank.value if left_rank else 'not found'}"
                left_suit_info = f"Suit: {left_suit.value if left_suit else 'not found'}"
                left_substep.match_info = f"{left_rank_info}, {left_suit_info}"

                if left_rank and left_suit:
                    left_card = Card(rank=left_rank, suit=left_suit)
                    left_substep.parsed_result = str(left_card)
                    left_substep.success = True
                    cards.append(left_card)
                else:
                    left_substep.parsed_result = "(incomplete detection)"
                    left_substep.success = False

            step.substeps.append(left_substep)

            # RIGHT CARD
            right_substep = DiagnosticStep(
                name="Right Card (hero_right)",
                description="Hero right card using hero_right library",
            )
            right_matcher = get_card_matcher(HERO_RIGHT_LIBRARY)

            right_rank_img = _extract_region(hero_region, right_rank_region)
            right_suit_img = _extract_region(hero_region, right_suit_region)
            right_substep.images.append(("Right Rank Region", right_rank_img))
            right_substep.images.append(("Right Suit Region", right_suit_img))

            # Check if slot is empty (matches table background)
            if bg_detector.is_empty_slot(right_rank_img):
                right_substep.match_info = "Empty slot (matches table background)"
                right_substep.parsed_result = "(empty slot)"
                right_substep.success = True
            else:
                right_rank = right_matcher.rank_matcher.match(right_rank_img)
                right_suit = right_matcher.suit_matcher.match(right_suit_img)

                right_rank_info = f"Rank: {right_rank.value if right_rank else 'not found'}"
                right_suit_info = f"Suit: {right_suit.value if right_suit else 'not found'}"
                right_substep.match_info = f"{right_rank_info}, {right_suit_info}"

                if right_rank and right_suit:
                    right_card = Card(rank=right_rank, suit=right_suit)
                    right_substep.parsed_result = str(right_card)
                    right_substep.success = True
                    cards.append(right_card)
                else:
                    right_substep.parsed_result = "(incomplete detection)"
                    right_substep.success = False

            step.substeps.append(right_substep)

            step.parsed_result = [str(c) for c in cards] if cards else []
            # Success if all substeps succeeded (even if both are empty slots)
            step.success = all(s.success for s in step.substeps)
            step.description += f" - Found {len(cards)} cards"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_players(self, frame: np.ndarray, region_detector) -> None:
        """Extract player info with diagnostics."""
        for position in self._PlayerPosition:
            step = DiagnosticStep(
                name=f"Player: {position.name}",
                description=f"Detecting player at {position.name} position",
            )

            try:
                player_regions = region_detector.get_player_region(position)

                # Chip detection
                from .fast_ocr import preprocess_for_ocr
                from .preprocessing import trim_to_content
                from .chip_ocr import extract_player_chips

                chip_substep = DiagnosticStep(
                    name="Chip Count",
                    description="Detecting chip count",
                )

                chip_region = region_detector.extract_player_chips(frame, position)
                chip_substep.images.append(("Chip Region", chip_region))

                trimmed = trim_to_content(chip_region)
                chip_substep.images.append(("Trimmed", trimmed))
                chip_substep.images.append(("OCR Input (inverted)", preprocess_for_ocr(trimmed)))

                amount = extract_player_chips(frame, region_detector, position)
                chip_substep.parsed_result = amount
                chip_substep.success = amount is not None

                step.substeps.append(chip_substep)

                # Note: Actions are deduced from chip changes, not detected from labels
                step.success = chip_substep.success

            except Exception as e:
                step.error = str(e)
                step.success = False

            self.report.steps.append(step)

    def _extract_player_names(self, frame: np.ndarray) -> None:
        """Extract player names with diagnostics."""
        from .name_detector import detect_player_names, extract_name_region

        step = DiagnosticStep(
            name="Player Name Detection",
            description="One-time detection of player names from name regions",
        )

        try:
            names = detect_player_names(frame, scale_frame=False)  # Already scaled

            for position in self._PlayerPosition:
                substep = DiagnosticStep(
                    name=f"{position.name}",
                    description=f"Name region for {position.name}",
                )

                # Extract and show name region image
                name_region = extract_name_region(frame, position, scale_frame=False)
                substep.images.append(("Name Region", name_region))

                detected_name = names.get(position)
                if detected_name:
                    substep.parsed_result = detected_name
                    substep.success = True
                else:
                    substep.parsed_result = "(not detected)"
                    substep.success = False

                step.substeps.append(substep)

            # Count successful detections
            detected_count = sum(1 for n in names.values() if n)
            step.parsed_result = {k.name: v for k, v in names.items() if v}
            step.description += f" - Detected {detected_count}/5 names"
            step.success = True  # Step succeeds even with partial detection

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)


def generate_html_report(report: DiagnosticReport, output_path: Optional[Path] = None) -> str:
    """
    Generate an HTML report from diagnostic data.

    Args:
        report: The diagnostic report to render.
        output_path: Optional path to write HTML file.

    Returns:
        HTML string.
    """
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
        h3 { color: #666; margin-top: 20px; }
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
        .image-container {
            text-align: center;
        }
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
        .parsed-result {
            font-weight: bold;
            color: #1976d2;
        }
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
        .frames-preview {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .frames-preview img {
            max-width: 600px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
"""]

    # Header
    html_parts.append(f"<h1>🎰 Frame Analysis Report</h1>")

    # Metadata
    html_parts.append('<div class="meta">')
    html_parts.append('<div class="meta-grid">')
    html_parts.append(f'<div class="meta-item">Generated: <span class="meta-value">{report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span></div>')
    if report.frame_number is not None:
        html_parts.append(f'<div class="meta-item">Frame #: <span class="meta-value">{report.frame_number}</span></div>')
    if report.frame_timestamp_ms is not None:
        html_parts.append(f'<div class="meta-item">Timestamp: <span class="meta-value">{report.frame_timestamp_ms/1000:.2f}s</span></div>')
    html_parts.append(f'<div class="meta-item">Original Size: <span class="meta-value">{report.original_size[0]}x{report.original_size[1]}</span></div>')
    html_parts.append(f'<div class="meta-item">Scaled Size: <span class="meta-value">{report.scaled_size[0]}x{report.scaled_size[1]}</span></div>')
    html_parts.append('</div></div>')

    # Frame previews
    html_parts.append('<h2>Frame Preview</h2>')
    html_parts.append('<div class="frames-preview">')
    if report.scaled_frame is not None:
        b64 = _image_to_base64(report.scaled_frame, max_width=700)
        html_parts.append(f'<div class="image-container"><img src="data:image/png;base64,{b64}"><div class="image-label">Scaled Frame (used for analysis)</div></div>')
    html_parts.append('</div>')

    # Steps
    html_parts.append('<h2>Detection Steps</h2>')

    for step in report.steps:
        html_parts.append(_render_step(step))

    html_parts.append("</body></html>")

    html = "\n".join(html_parts)

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html)

    return html


def _render_step(step: DiagnosticStep, level: int = 0) -> str:
    """Render a single step as HTML."""
    status_class = "success" if step.success else "failure"
    status_text = "✓ Success" if step.success else "✗ Failed"

    html = f'<div class="step {status_class}">'
    html += '<div class="step-header">'
    html += f'<span class="step-name">{step.name}</span>'
    html += f'<span class="badge {status_class}">{status_text}</span>'
    html += '</div>'

    html += f'<div class="description">{step.description}</div>'

    # Images
    if step.images:
        html += '<div class="images">'
        for label, img in step.images:
            if img is not None and img.size > 0:
                b64 = _image_to_base64(img)
                html += f'<div class="image-container">'
                html += f'<img src="data:image/png;base64,{b64}">'
                html += f'<div class="image-label">{label}</div>'
                html += '</div>'
        html += '</div>'

    # OCR result (for pot/chips)
    if step.ocr_result is not None:
        html += f'<div class="ocr-result">OCR Result: <code>{step.ocr_result}</code></div>'

    # Match info (for card library matching)
    if step.match_info is not None:
        html += f'<div class="ocr-result">Match: <code>{step.match_info}</code></div>'

    # Parsed result
    if step.parsed_result is not None:
        html += f'<div class="parsed-result">Parsed: {step.parsed_result}</div>'

    # Error
    if step.error:
        html += f'<div class="error">Error: {step.error}</div>'

    # Substeps
    if step.substeps:
        html += '<div class="substeps">'
        for substep in step.substeps:
            html += _render_step(substep, level + 1)
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
        frame_number: Frame number to analyze (mutually exclusive with timestamp_s).
        timestamp_s: Timestamp in seconds (mutually exclusive with frame_number).

    Returns:
        Dictionary with report statistics.
    """
    from .frame_extractor import VideoFrameExtractor

    with VideoFrameExtractor(video_path) as video:
        # Determine which frame to analyze
        if frame_number is not None:
            frame_info = video.get_frame_at(frame_number)
        elif timestamp_s is not None:
            frame_info = video.get_frame_at_timestamp(timestamp_s * 1000)
        else:
            frame_info = video.get_frame_at(0)

        if frame_info is None:
            raise ValueError("Could not read frame")

        # Run diagnostic extraction
        extractor = DiagnosticExtractor()
        report = extractor.extract_with_diagnostics(
            frame_info.image,
            frame_number=frame_info.frame_number,
            timestamp_ms=frame_info.timestamp_ms,
        )

        # Generate HTML report
        generate_html_report(report, output_path)

        # Return statistics
        successes = sum(1 for s in report.steps if s.success)
        failures = sum(1 for s in report.steps if not s.success)

        return {
            "frame_number": frame_info.frame_number,
            "timestamp_ms": frame_info.timestamp_ms,
            "steps_succeeded": successes,
            "steps_failed": failures,
            "output_path": output_path,
        }
