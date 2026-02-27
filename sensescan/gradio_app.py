from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from loguru import logger

from .config import DATA_DIR, setup_logging
from .pipeline import run_handwritten_pipeline, segments_to_text


setup_logging()


def _create_request_dir(original_filename: str | None) -> Path:
    """Create a unique per-request directory under the global data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(original_filename or "request").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "request"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    short_id = uuid.uuid4().hex[:8]

    run_dir = DATA_DIR / f"{safe_stem}_{ts}_{short_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_artifacts(
    request_dir: Path,
    original_bytes: bytes,
    text: str,
    segments: Dict[str, object],
    timings: Dict[str, float],
) -> None:
    """Persist input image and OCR outputs into the per-request directory."""
    try:
        # Store the input as PNG, since we already encoded it that way
        (request_dir / "input.png").write_bytes(original_bytes)

        (request_dir / "ocr.txt").write_text(text, encoding="utf-8")

        with (request_dir / "ocr.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"text": text, "segments": segments, "timings": timings},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - logging only
        logger.exception(f"Failed to persist artifacts for Gradio request: {e}")


def _run_sensescan(
    image: np.ndarray,
) -> Tuple[str, Dict[str, float]]:
    """
    Core Gradio callback.

    - image: RGB numpy array from Gradio.
    - image_name: original filename (if available).
    """
    if image is None:
        return "", {}

    request_dir = _create_request_dir(None)

    # Encode back to bytes so we can persist the original upload.
    # Use PNG for lossless storage if original bytes are not directly available.
    success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        return "", {}, {}, "Failed to encode image."
    original_bytes = buffer.tobytes()

    # Convert to BGR for pipeline
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    logger.info(f"SenseScan Gradio: shape={bgr.shape}, dir={request_dir}")

    segments, timings = run_handwritten_pipeline(bgr)
    text = segments_to_text(segments)

    _save_artifacts(request_dir, original_bytes, text, segments, timings)

    return text, timings


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="SenseScan – Handwritten OCR") as demo:
        gr.Markdown(
            "## SenseScan – Handwritten Document OCR\n"
            "Upload a handwritten page on the left, then run SenseScan to view "
            "plain-text OCR, structured segments, and timing information on the right."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Handwritten page",
                    type="numpy",
                    image_mode="RGB",
                )
                run_button = gr.Button("Run SenseScan OCR", variant="primary")

            with gr.Column(scale=1):
                with gr.Tab("Text"):
                    text_output = gr.Textbox(
                        label="Recognized text",
                        lines=12,
                        show_copy_button=True,
                    )
                with gr.Tab("Timings"):
                    timings_output = gr.JSON(label="Timing breakdown (seconds)")

        run_button.click(
            fn=_run_sensescan,
            inputs=[image_input],
            outputs=[text_output, timings_output],
            concurrency_limit=4,
            queue=True,
        )

    return demo


def main() -> None:
    demo = build_interface()

    # Stable internal URL; for a public, shareable URL, set
    # SENSESCAN_GRADIO_SHARE=true in the environment.
    share = os.environ.get("SENSESCAN_GRADIO_SHARE", "false").lower() == "true"
    port = int(os.environ.get("SENSESCAN_GRADIO_PORT", "8002"))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        max_threads=4,
    )


if __name__ == "__main__":
    main()

