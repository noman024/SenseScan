from __future__ import annotations

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel, Field
import uvicorn

from .config import DATA_DIR, setup_logging
from .pipeline import run_handwritten_pipeline, segments_to_text


setup_logging()

app = FastAPI(
    title="SenseScan – Handwritten Document OCR API",
    description=(
        "SenseScan is a focused handwritten document OCR service for full-page handwritten images. "
        "It provides a simple plain-text endpoint and a richer JSON API for applications "
        "that need layout and timing information."
    ),
)


class Point(BaseModel):
    x: float
    y: float


class TextSegment(BaseModel):
    id: str = Field(..., description="Stable identifier of the text segment")
    text: str = Field(..., description="Recognized text for this segment")
    points: List[Point] = Field(
        ..., description="Quadrilateral of the segment in image coordinates (x, y)"
    )


class TimingInfo(BaseModel):
    total_segment_time: Optional[float] = Field(
        None, description="Time spent in segmentation (seconds)"
    )
    word_rec_time: Optional[float] = Field(
        None, description="Time spent in word recognition (seconds)"
    )
    pipeline_time: Optional[float] = Field(
        None, description="End-to-end API time including I/O (seconds)"
    )


class HandwrittenOCRResponse(BaseModel):
    text: str = Field(..., description="Reconstructed page text in reading order")
    segments: Optional[List[TextSegment]] = Field(
        None,
        description="Per-segment text and geometry if requested via `include_segments=true`",
    )
    timings: Optional[TimingInfo] = Field(
        None,
        description="Timing breakdown if requested via `include_timings=true`",
    )


def _create_request_dir(original_filename: Optional[str]) -> Path:
    """Create a unique per-request directory under the global data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(original_filename or "request").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "request"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    short_id = uuid.uuid4().hex[:8]

    run_dir = DATA_DIR / f"{safe_stem}_{ts}_{short_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_image_from_upload(image: UploadFile, save_dir: Path) -> cv2.Mat:
    """Load an uploaded image, persist the original file, and decode into BGR."""
    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # Persist original upload
    original_name = image.filename or "input"
    suffix = Path(original_name).suffix or ".bin"
    input_path = save_dir / f"input{suffix}"
    input_path.write_bytes(data)

    # Decode into OpenCV image
    np_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    return img


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok", "service": "sensescan-handwritten-ocr"}


@app.post("/plugin", response_class=PlainTextResponse)
async def handwritten_plugin(image: UploadFile = File(...)) -> str:
    """
    Plain-text SenseScan handwritten OCR endpoint.

    - Accepts an uploaded image (multipart/form-data, field name: ``image``).
    - Runs EAST-based segmentation + CRNN word recognition (handwritten Bangla).
    - Returns only the reconstructed, reading-order text as **plain text** (no JSON wrapper).
    """
    start_api_time = time.time()
    timings: Dict[str, float] = {}

    request_dir = _create_request_dir(image.filename)
    img = _load_image_from_upload(image, request_dir)
    logger.info(
        f"SenseScan /plugin: file={image.filename}, shape={img.shape}, dir={request_dir}"
    )

    segments, ocr_timings = run_handwritten_pipeline(img)
    timings.update(ocr_timings)

    text = segments_to_text(segments)

    timings["pipeline_time"] = round(time.time() - start_api_time, 4)
    logger.info(f"SenseScan /plugin timings: {timings}")

    # Persist outputs
    try:
        (request_dir / "ocr.txt").write_text(text, encoding="utf-8")
        import json

        with (request_dir / "ocr.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"text": text, "segments": segments, "timings": timings},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - logging only
        logger.exception(f"Failed to persist outputs for /plugin request: {e}")

    return text


@app.post(
    "/v1/ocr/handwritten",
    response_model=HandwrittenOCRResponse,
    summary="Handwritten page OCR (JSON)",
)
async def handwritten_ocr_v1(
    image: UploadFile = File(..., description="Handwritten page image"),
    include_segments: bool = Query(
        False,
        description="If true, include per-segment text and geometry in the response",
    ),
    include_timings: bool = Query(
        False,
        description="If true, include a simple timing breakdown in the response",
    ),
) -> HandwrittenOCRResponse:
    """
    Primary SenseScan JSON API for handwritten document OCR.

    **Request**
    - Content type: ``multipart/form-data``
    - Fields:
      - ``image``: uploaded image file (handwritten page)
    - Query parameters:
      - ``include_segments``: return per-word segments with coordinates
      - ``include_timings``: return simple timing breakdown

    **Response**
    - JSON body with:
      - ``text``: reconstructed page text in reading order
      - optional ``segments`` and ``timings`` depending on flags
    """
    start_api_time = time.time()
    timings: Dict[str, float] = {}

    request_dir = _create_request_dir(image.filename)
    img = _load_image_from_upload(image, request_dir)
    logger.info(
        f"SenseScan /v1/ocr/handwritten: file={image.filename}, shape={img.shape}, dir={request_dir}"
    )

    segments_dict, ocr_timings = run_handwritten_pipeline(img)
    timings.update(ocr_timings)

    full_text = segments_to_text(segments_dict)

    segments_out: Optional[List[TextSegment]] = None
    if include_segments:
        segments_out = []
        for seg_id, seg in segments_dict.items():
            seg_points = seg.get("points", [])  # type: ignore[assignment]
            text = seg.get("text", {}).get("0", "")  # type: ignore[assignment]
            if not text:
                continue
            points = [Point(x=float(p["x"]), y=float(p["y"])) for p in seg_points]
            segments_out.append(TextSegment(id=str(seg_id), text=text, points=points))

    timings_out: Optional[TimingInfo] = None
    if include_timings:
        timings["pipeline_time"] = round(time.time() - start_api_time, 4)
        timings_out = TimingInfo(
            total_segment_time=timings.get("total_segment_time"),
            word_rec_time=timings.get("word_rec_time"),
            pipeline_time=timings.get("pipeline_time"),
        )
        logger.info(f"SenseScan /v1/ocr/handwritten timings: {timings}")

    response = HandwrittenOCRResponse(
        text=full_text,
        segments=segments_out,
        timings=timings_out,
    )

    # Persist outputs
    try:
        (request_dir / "ocr.txt").write_text(full_text, encoding="utf-8")
        import json

        with (request_dir / "ocr.json").open("w", encoding="utf-8") as f:
            json.dump(
                response.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - logging only
        logger.exception(f"Failed to persist outputs for /v1/ocr/handwritten request: {e}")

    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("sensescan.api:app", host="0.0.0.0", port=port, reload=False)

