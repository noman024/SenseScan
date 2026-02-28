from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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
    request_dir: Optional[Path] = None,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Run the full handwritten OCR pipeline and return:
    - segments_dict: mapping from UUID to text block metadata
    - timings: simple timing breakdown for segmentation and recognition

    If request_dir is provided, intermediate outputs are saved:
    preprocessed.png, detection_boxes.json, detection_boxes.png, crops/*.png,
    crops/*_model_input.png, and pipeline_log.txt (plus ocr.txt/ocr.json by the API/Gradio).
    """
    h, w = image.shape[:2]
    logger.info(
        "pipeline start | image_shape=({}, {}), request_dir={}",
        h, w, request_dir,
    )

    timings: Dict[str, float] = {}
    save_preprocessed = (request_dir / "preprocessed.png") if request_dir is not None else None

    logger.info("pipeline step=detection start")
    start_time = time.time()
    point_list, roi_list = _east.detect_words(image, save_preprocessed_path=save_preprocessed)
    timings["total_segment_time"] = round(time.time() - start_time, 4)
    logger.info(
        "pipeline step=detection done | boxes={}, time_s={}",
        len(roi_list), timings["total_segment_time"],
    )

    if request_dir is not None:
        try:
            with (request_dir / "detection_boxes.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"points": point_list, "rois": roi_list},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            vis = image.copy()
            for roi in roi_list:
                x_min, x_max, y_min, y_max = roi
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imwrite(str(request_dir / "detection_boxes.png"), vis)
            logger.info("pipeline detection outputs saved | dir={}", request_dir)
        except Exception as e:  # pragma: no cover - logging only
            logger.warning("pipeline detection save failed | error={}", e)

    logger.info("pipeline step=recognition start | rois={}", len(roi_list))
    start_time = time.time()
    recognized_words_list: List[Dict[str, object]] = []
    if len(roi_list) > 0:
        recognized_words_list = _word_recog.infer(
            image, roi_list, request_dir=request_dir
        )
    timings["word_rec_time"] = round(time.time() - start_time, 4)
    logger.info(
        "pipeline step=recognition done | words={}, time_s={}",
        len(recognized_words_list), timings["word_rec_time"],
    )

    logger.info(
        "pipeline step=segment_build | building textline_dict from points={}, words={}",
        len(point_list), len(recognized_words_list),
    )

    textline_dict: Dict[str, object] = {}
    for word in recognized_words_list:
        idx = int(word["id"]) - 1
        if idx < 0 or idx >= len(point_list):
            continue
        text = (word.get("text") or "").strip()
        if not text:
            continue
        point = point_list[idx]
        unique_id = str(uuid.uuid1())
        sentence_dict = {
            "id": unique_id,
            "type": "TextBlock",
            "points": point,
            "orientation": None,
            "isRelative": False,
            "text": {"0": text},
            "readingOrder": [],
            "relative": False,
            "textstyle": None,
        }
        textline_dict[unique_id] = sentence_dict

    logger.info(
        "pipeline done | segments={}, total_segment_time_s={}, word_rec_time_s={}",
        len(textline_dict), timings["total_segment_time"], timings["word_rec_time"],
    )

    if request_dir is not None:
        try:
            log_lines = [
                f"Pipeline run at {datetime.now(timezone.utc).isoformat()}",
                f"image_shape: {h} x {w}",
                f"detection_boxes: {len(roi_list)}",
                f"recognized_words: {len(recognized_words_list)}",
                f"segments: {len(textline_dict)}",
                f"total_segment_time_s: {timings.get('total_segment_time', 0)}",
                f"word_rec_time_s: {timings.get('word_rec_time', 0)}",
                "status: completed",
            ]
            (request_dir / "pipeline_log.txt").write_text(
                "\n".join(log_lines), encoding="utf-8"
            )
            logger.info("pipeline run log saved | dir={}", request_dir)
        except Exception as e:  # pragma: no cover - logging only
            logger.warning("pipeline run log save failed | error={}", e)

    return textline_dict, timings


def segments_to_text(segments: Dict[str, object]) -> str:
    """Reconstruct text by grouping segments into lines and sorting within each line."""
    logger.debug("segments_to_text | input_segments={}", len(segments))
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

    out = "\n".join(line_texts)
    logger.debug("segments_to_text | lines={} chars={}", len(line_texts), len(out))
    return out

