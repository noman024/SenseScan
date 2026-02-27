from __future__ import annotations

import time
import uuid
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from .config import EAST_MODEL_PATH, WORD_MODEL_PATH, get_runtime_config, setup_logging
from .detection import EASTSegmentation
from .recognition import HandwrittenWordRecognizer


setup_logging()

_runtime = get_runtime_config()
_east = EASTSegmentation(EAST_MODEL_PATH, _runtime.device, num_gpus=_runtime.num_gpus)
_word_recog = HandwrittenWordRecognizer(
    WORD_MODEL_PATH, _runtime.device, num_gpus=_runtime.num_gpus
)


def run_handwritten_pipeline(
    image: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Run the full handwritten OCR pipeline and return:
    - segments_dict: mapping from UUID to text block metadata
    - timings: simple timing breakdown for segmentation and recognition
    """
    timings: Dict[str, float] = {}

    start_time = time.time()
    point_list, roi_list = _east.detect_words(image)
    timings["total_segment_time"] = round(time.time() - start_time, 4)

    start_time = time.time()
    recognized_words_list: List[Dict[str, object]] = []
    if len(roi_list) > 0:
        recognized_words_list = _word_recog.infer(image, roi_list)
    timings["word_rec_time"] = round(time.time() - start_time, 4)

    logger.info(
        "SenseScan pipeline: detected_boxes=%d, recognized_words=%d",
        len(roi_list),
        len(recognized_words_list),
    )

    textline_dict: Dict[str, object] = {}
    for word in recognized_words_list:
        idx = int(word["id"]) - 1
        if idx < 0 or idx >= len(point_list):
            continue
        point = point_list[idx]
        unique_id = str(uuid.uuid1())
        sentence_dict = {
            "id": unique_id,
            "type": "TextBlock",
            "points": point,
            "orientation": None,
            "isRelative": False,
            "text": {"0": word["text"]},
            "readingOrder": [],
            "relative": False,
            "textstyle": None,
        }
        textline_dict[unique_id] = sentence_dict

    return textline_dict, timings


def segments_to_text(segments: Dict[str, object]) -> str:
    """Reconstruct text by grouping segments into lines and sorting within each line."""
    boxes: List[Dict[str, float]] = []
    for _, seg in segments.items():
        seg_dict = seg  # type: ignore[assignment]
        pts = seg_dict.get("points", [])  # type: ignore[assignment]
        if len(pts) != 4:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = 0.5 * (y_min + y_max)
        height = y_max - y_min + 1
        text = seg_dict.get("text", {}).get("0", "").strip()  # type: ignore[assignment]
        if not text:
            continue
        boxes.append(
            {
                "y_center": y_center,
                "height": height,
                "x_min": x_min,
                "x_max": x_max,
                "text": text,
            }
        )

    if not boxes:
        return ""

    boxes.sort(key=lambda b: b["y_center"])

    lines: List[List[Dict[str, float]]] = []
    current_line: List[Dict[str, float]] = []
    line_merge_factor = 0.6

    for b in boxes:
        if not current_line:
            current_line.append(b)
            continue

        avg_y = sum(bb["y_center"] for bb in current_line) / len(current_line)
        avg_h = sum(bb["height"] for bb in current_line) / len(current_line)
        if abs(b["y_center"] - avg_y) <= line_merge_factor * avg_h:
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]

    if current_line:
        lines.append(current_line)

    line_texts: List[str] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: b["x_min"])
        words = [b["text"] for b in line_sorted]
        line_texts.append(" ".join(words))

    return "\n".join(line_texts)

