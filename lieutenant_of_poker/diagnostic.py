"""
Diagnostic report generator for frame analysis.

Creates detailed HTML reports showing each step of the detection pipeline
with images, intermediate results, and OCR outputs.
"""

import base64
import io
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image


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


def diagnose(
    video_path: str,
    output: str = "diagnostic_report.html",
    frame_number: Optional[int] = None,
    timestamp_s: Optional[float] = None,
    open_browser: bool = False,
) -> dict:
    """
    Generate a diagnostic HTML report for a specific frame.

    Args:
        video_path: Path to the video file.
        output: Output path for the HTML report.
        frame_number: Frame number to analyze (takes precedence over timestamp_s).
        timestamp_s: Timestamp in seconds.
        open_browser: Whether to open the report in a browser.

    Returns:
        Dictionary with report statistics.
    """
    from .first_frame import TableInfo
    from .frame_extractor import VideoFrameExtractor, get_video_info
    from .card_matcher import (
        COMMUNITY_LIBRARY, HERO_LIBRARY, _match_rank, _match_suit,
        extract_community_card_regions, match_community_cards,
        HERO_RANK_OFFSET, HERO_SUIT_OFFSET, HERO_RANK_SIZE, HERO_SUIT_SIZE,
        HERO_CARD_SPACING, match_hero_cards,
    )
    from .chip_ocr import get_money_region, extract_player_money, extract_pot, _POT_POS
    from .fast_ocr import preprocess_for_ocr

    # Print video info
    info = get_video_info(video_path)
    print(f"Video: {video_path}", file=sys.stderr)
    print(f"  Resolution: {info['width']}x{info['height']}", file=sys.stderr)
    print(f"  Duration: {info['duration_seconds']:.1f}s", file=sys.stderr)

    table = TableInfo.from_video(video_path)
    output_path = Path(output)
    steps = []

    with VideoFrameExtractor(video_path) as video:
        if frame_number is not None:
            frame_info = video.get_frame_at(frame_number)
        else:
            frame_info = video.get_frame_at_timestamp((timestamp_s or 0) * 1000)

        if frame_info is None:
            raise ValueError("Could not read frame")

        frame = frame_info.image

        # --- Community Cards ---
        community_step = {
            "name": "Community Cards Detection",
            "description": "Detecting cards on the board",
            "substeps": [],
            "success": True,
        }
        try:
            regions = extract_community_card_regions(frame)
            matched_cards = match_community_cards(frame)

            cards = []
            for i, (rank_img, suit_img) in enumerate(regions.cards):
                rank = _match_rank(rank_img, COMMUNITY_LIBRARY)
                suit = _match_suit(suit_img, COMMUNITY_LIBRARY)
                card = matched_cards[i]

                substep = {
                    "name": f"Slot {i+1}",
                    "description": f"Card slot {i+1} of 5",
                    "images": [
                        ("Rank Region", _image_to_base64(rank_img)),
                        ("Suit Region", _image_to_base64(suit_img)),
                    ],
                    "match_info": f"Rank: {rank or 'not found'}, Suit: {suit or 'not found'}",
                    "parsed_result": card if card else "(not detected)",
                    "success": True,
                }
                if card:
                    cards.append(card)
                community_step["substeps"].append(substep)

            community_step["parsed_result"] = cards if cards else []
            community_step["description"] += f" - Found {len(cards)} cards"
        except Exception as e:
            community_step["error"] = str(e)
            community_step["success"] = False
        steps.append(community_step)

        # --- Pot ---
        pot_step = {
            "name": "Pot Detection",
            "description": "Extracting pot amount",
            "success": True,
        }
        try:
            pot_region = get_money_region(frame, _POT_POS, no_currency=table.no_currency)
            pot_amount = extract_pot(frame, no_currency=table.no_currency)
            pot_step["images"] = [
                ("Pot Region", _image_to_base64(pot_region)),
                ("OCR Input", _image_to_base64(preprocess_for_ocr(pot_region))),
            ]
            pot_step["ocr_result"] = str(pot_amount) if pot_amount is not None else "(not detected)"
            pot_step["parsed_result"] = pot_amount
            pot_step["success"] = pot_amount is not None
        except Exception as e:
            pot_step["error"] = str(e)
            pot_step["success"] = False
        steps.append(pot_step)

        # --- Hero Cards ---
        hero_position = table.positions[-1] if table.positions else None
        if hero_position:
            hero_step = {
                "name": "Hero Cards Detection",
                "description": f"Detecting hero's hole cards at position {hero_position}",
                "substeps": [],
                "success": True,
            }
            try:
                px, py = hero_position
                left_rank_region = (px + HERO_RANK_OFFSET[0], py + HERO_RANK_OFFSET[1], *HERO_RANK_SIZE)
                left_suit_region = (px + HERO_SUIT_OFFSET[0], py + HERO_SUIT_OFFSET[1], *HERO_SUIT_SIZE)

                def extract_region(r):
                    x, y, w, h = r
                    return frame[y:y+h, x:x+w]

                left_rank_img = extract_region(left_rank_region)
                left_suit_img = extract_region(left_suit_region)
                right_rank_img = extract_region((left_rank_region[0] + HERO_CARD_SPACING, *left_rank_region[1:]))
                right_suit_img = extract_region((left_suit_region[0] + HERO_CARD_SPACING, *left_suit_region[1:]))

                matched_cards = match_hero_cards(frame, hero_position)
                cards = []

                # Left card
                left_rank = _match_rank(left_rank_img, HERO_LIBRARY)
                left_suit = _match_suit(left_suit_img, HERO_LIBRARY)
                left_card = matched_cards[0]
                left_substep = {
                    "name": "Left Card",
                    "description": "Hero left card",
                    "images": [
                        ("Rank Region", _image_to_base64(left_rank_img)),
                        ("Suit Region", _image_to_base64(left_suit_img)),
                    ],
                    "match_info": f"Rank: {left_rank or 'not found'}, Suit: {left_suit or 'not found'}",
                    "parsed_result": left_card if left_card else "(not detected)",
                    "success": left_card is not None,
                }
                if left_card:
                    cards.append(left_card)
                hero_step["substeps"].append(left_substep)

                # Right card
                right_rank = _match_rank(right_rank_img, HERO_LIBRARY)
                right_suit = _match_suit(right_suit_img, HERO_LIBRARY)
                right_card = matched_cards[1]
                right_substep = {
                    "name": "Right Card",
                    "description": "Hero right card",
                    "images": [
                        ("Rank Region", _image_to_base64(right_rank_img)),
                        ("Suit Region", _image_to_base64(right_suit_img)),
                    ],
                    "match_info": f"Rank: {right_rank or 'not found'}, Suit: {right_suit or 'not found'}",
                    "parsed_result": right_card if right_card else "(not detected)",
                    "success": right_card is not None,
                }
                if right_card:
                    cards.append(right_card)
                hero_step["substeps"].append(right_substep)

                hero_step["parsed_result"] = cards if cards else []
                hero_step["success"] = all(s["success"] for s in hero_step["substeps"])
                hero_step["description"] += f" - Found {len(cards)} cards"
            except Exception as e:
                hero_step["error"] = str(e)
                hero_step["success"] = False
            steps.append(hero_step)

        # --- Player Money ---
        if table.positions:
            money_step = {
                "name": "Player Money Detection",
                "description": f"Extracting money for {len(table.positions)} players",
                "substeps": [],
                "success": True,
            }
            try:
                results = {}
                for i in range(len(table.positions)):
                    name = table.names[i] if i < len(table.names) else f"Player {i}"
                    pos = table.positions[i]

                    money_region = get_money_region(frame, pos, no_currency=table.no_currency)
                    amount = extract_player_money(frame, table, i)

                    substep = {
                        "name": name,
                        "description": f"Position {i}: ({pos[0]}, {pos[1]})",
                        "images": [
                            ("Money Region", _image_to_base64(money_region)),
                            ("OCR Input", _image_to_base64(preprocess_for_ocr(money_region))),
                        ],
                        "ocr_result": str(amount) if amount is not None else "(not detected)",
                        "parsed_result": amount,
                        "success": amount is not None,
                    }

                    if amount is not None:
                        results[name] = amount

                    money_step["substeps"].append(substep)

                money_step["parsed_result"] = results
                money_step["success"] = len(results) > 0
                money_step["description"] += f" - Found {len(results)} amounts"
            except Exception as e:
                money_step["error"] = str(e)
                money_step["success"] = False
            steps.append(money_step)

        # --- Hero Detection (for splitter) ---
        from ._detection import detect_hero_cards
        from .fast_ocr import ocr_name_at_position, get_name_region
        from ._positions import SEAT_POSITIONS
        detection_step = {
            "name": "Hero Detection (Splitter)",
            "description": "Would this frame be recognized as active by the video splitter?",
            "substeps": [],
            "success": True,
        }
        try:
            is_active = detect_hero_cards(frame)
            detection_step["parsed_result"] = "YES - Hero detected" if is_active else "NO - Hero not detected"
            detection_step["success"] = is_active

            # Show name region for all seat positions
            for i, pos in enumerate(SEAT_POSITIONS):
                name_region = get_name_region(frame, pos)
                detected_name = ocr_name_at_position(frame, pos)

                substep = {
                    "name": f"Seat {i}",
                    "description": f"Position ({pos[0]}, {pos[1]})",
                    "parsed_result": detected_name if detected_name else "(not detected)",
                    "success": detected_name is not None,
                }
                if name_region is not None:
                    substep["images"] = [
                        ("Name Region", _image_to_base64(name_region)),
                        ("OCR Input", _image_to_base64(preprocess_for_ocr(name_region))),
                    ]
                detection_step["substeps"].append(substep)
        except Exception as e:
            detection_step["error"] = str(e)
            detection_step["success"] = False
        steps.append(detection_step)

        # --- Render HTML ---
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("diagnostic_report.html")

        html = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_number=frame_info.frame_number,
            timestamp_ms=frame_info.timestamp_ms,
            frame_size=(frame.shape[1], frame.shape[0]),
            frame_b64=_image_to_base64(frame, max_width=700),
            steps=steps,
        )

        output_path.write_text(html)

        successes = sum(1 for s in steps if s["success"])
        failures = sum(1 for s in steps if not s["success"])

        print(f"\nAnalyzing frame {frame_info.frame_number} ({frame_info.timestamp_ms/1000:.2f}s)...", file=sys.stderr)
        print(f"Report generated: {output_path}", file=sys.stderr)
        print(f"  Steps: {successes} succeeded, {failures} failed", file=sys.stderr)

        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")

        return {
            "frame_number": frame_info.frame_number,
            "timestamp_ms": frame_info.timestamp_ms,
            "hero_position": hero_position,
            "steps_succeeded": successes,
            "steps_failed": failures,
            "output_path": output_path,
        }
