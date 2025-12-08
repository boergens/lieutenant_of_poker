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

from lieutenant_of_poker.table_regions import detect_table_regions, NUM_PLAYERS, seat_name


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
    frame_size: tuple[int, int] = (0, 0)
    frame: Optional[np.ndarray] = None
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
        self._detect_regions = detect_table_regions
        self._num_players = NUM_PLAYERS
        self._seat_name = seat_name
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
            frame_size=(frame.shape[1], frame.shape[0]),
            frame=frame.copy(),
        )

        # Create region detector
        region_detector = self._detect_regions(frame)

        # Extract each component with diagnostics
        self._extract_pot(frame, region_detector)
        self._extract_community_cards(frame, region_detector)
        self._extract_hero_cards(frame, region_detector)
        self._extract_players(frame, region_detector)
        self._extract_player_names(frame)
        self._extract_dealer_button(frame, region_detector)

        return self.report

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
        """Extract community cards with diagnostics using absolute frame coordinates."""
        step = DiagnosticStep(
            name="Community Cards Detection",
            description="Detecting cards on the board using absolute coordinates",
        )

        try:
            from .card_matcher import (
                COMMUNITY_LIBRARY, _match_rank, _match_suit,
                extract_community_card_regions, match_community_cards,
            )

            # Extract regions for diagnostic display
            regions = extract_community_card_regions(frame)

            # Get matched cards
            matched_cards = match_community_cards(frame)

            cards = []
            for i, (rank_img, suit_img) in enumerate(regions.cards):
                substep = DiagnosticStep(
                    name=f"Slot {i+1}",
                    description=f"Card slot {i+1} of 5",
                )
                substep.images.append(("Rank Region", rank_img))
                substep.images.append(("Suit Region", suit_img))

                # Match rank and suit
                rank = _match_rank(rank_img, COMMUNITY_LIBRARY)
                suit = _match_suit(suit_img, COMMUNITY_LIBRARY)

                rank_info = f"Rank: {rank if rank else 'not found'}"
                suit_info = f"Suit: {suit if suit else 'not found'}"
                substep.match_info = f"{rank_info}, {suit_info}"

                card = matched_cards[i]
                if card:
                    substep.parsed_result = card
                    substep.success = True
                    cards.append(card)
                else:
                    substep.parsed_result = "(not detected)"
                    substep.success = True  # Empty is a valid state (preflop/flop)

                step.substeps.append(substep)

            step.parsed_result = cards if cards else []
            step.success = True  # Success even with 0 cards (preflop)
            step.description += f" - Found {len(cards)} cards"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_hero_cards(self, frame: np.ndarray, region_detector) -> None:
        """Extract hero cards with diagnostics using absolute frame coordinates."""
        step = DiagnosticStep(
            name="Hero Cards Detection",
            description="Detecting hero's hole cards",
        )

        try:
            from .card_matcher import (
                HERO_LIBRARY, _match_rank, _match_suit,
                extract_hero_card_regions, match_hero_cards,
            )

            # Extract regions directly from frame
            regions = extract_hero_card_regions(frame)

            # Get matched cards
            matched_cards = match_hero_cards(frame)

            cards = []

            # LEFT CARD
            left_substep = DiagnosticStep(
                name="Left Card",
                description="Hero left card",
            )
            left_substep.images.append(("Left Rank Region", regions.left_rank))
            left_substep.images.append(("Left Suit Region", regions.left_suit))

            left_rank = _match_rank(regions.left_rank, HERO_LIBRARY)
            left_suit = _match_suit(regions.left_suit, HERO_LIBRARY)

            left_rank_info = f"Rank: {left_rank if left_rank else 'not found'}"
            left_suit_info = f"Suit: {left_suit if left_suit else 'not found'}"
            left_substep.match_info = f"{left_rank_info}, {left_suit_info}"

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
            right_substep = DiagnosticStep(
                name="Right Card",
                description="Hero right card",
            )
            right_substep.images.append(("Right Rank Region", regions.right_rank))
            right_substep.images.append(("Right Suit Region", regions.right_suit))

            right_rank = _match_rank(regions.right_rank, HERO_LIBRARY)
            right_suit = _match_suit(regions.right_suit, HERO_LIBRARY)

            right_rank_info = f"Rank: {right_rank if right_rank else 'not found'}"
            right_suit_info = f"Suit: {right_suit if right_suit else 'not found'}"
            right_substep.match_info = f"{right_rank_info}, {right_suit_info}"

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
            # Success if all substeps succeeded
            step.success = all(s.success for s in step.substeps)
            step.description += f" - Found {len(cards)} cards"

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_players(self, frame: np.ndarray, region_detector) -> None:
        """Extract player info with diagnostics."""
        for position in range(self._num_players):
            step = DiagnosticStep(
                name=f"Player: {self._seat_name(position)}",
                description=f"Detecting player at {self._seat_name(position)} position",
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

            for position in range(self._num_players):
                substep = DiagnosticStep(
                    name=self._seat_name(position),
                    description=f"Name region for {self._seat_name(position)}",
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
            step.parsed_result = {self._seat_name(k): v for k, v in names.items() if v}
            step.description += f" - Detected {detected_count}/5 names"
            step.success = True  # Step succeeds even with partial detection

        except Exception as e:
            step.error = str(e)
            step.success = False

        self.report.steps.append(step)

    def _extract_dealer_button(self, frame: np.ndarray, region_detector) -> None:
        """Extract dealer button position with diagnostics."""
        from .dealer_detector import detect_dealer_button, detect_dealer_position

        step = DiagnosticStep(
            name="Dealer Button Detection",
            description="Detecting dealer button position using template matching",
        )

        try:
            # Get the search region
            search_region = region_detector.dealer_button_search_region
            search_img = search_region.extract(frame)
            step.images.append(("Search Region", search_img))

            # Detect button position
            button_pos = detect_dealer_button(frame, region_detector)
            if button_pos is not None:
                # Draw marker on search region
                marked = search_img.copy()
                rel_x = button_pos[0] - search_region.x
                rel_y = button_pos[1] - search_region.y
                cv2.circle(marked, (rel_x, rel_y), 20, (0, 255, 0), 2)
                cv2.circle(marked, (rel_x, rel_y), 3, (0, 255, 0), -1)
                step.images.append(("Detected Location", marked))

                # Determine which player
                dealer_seat = detect_dealer_position(frame, region_detector)
                if dealer_seat is not None:
                    step.parsed_result = f"Button at {self._seat_name(dealer_seat)} (pixel: {button_pos})"
                else:
                    step.parsed_result = f"Button found at {button_pos} but no player match"
                step.success = True
            else:
                step.parsed_result = "(not detected)"
                step.success = False

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
    html_parts.append(f'<div class="meta-item">Frame Size: <span class="meta-value">{report.frame_size[0]}x{report.frame_size[1]}</span></div>')
    html_parts.append('</div></div>')

    # Frame preview
    html_parts.append('<h2>Frame Preview</h2>')
    html_parts.append('<div class="frames-preview">')
    if report.frame is not None:
        b64 = _image_to_base64(report.frame, max_width=700)
        html_parts.append(f'<div class="image-container"><img src="data:image/png;base64,{b64}"><div class="image-label">Frame</div></div>')
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
