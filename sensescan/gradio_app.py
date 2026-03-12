from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        logger.exception("gradio save_artifacts failed | error={}", e)


def _run_sensescan(
    image: np.ndarray,
) -> Tuple[
    str,
    Optional[np.ndarray],
    Dict[str, float],
    Optional[str],
    str,
    str,
    str,
]:
    """
    Core Gradio callback.

    - image: RGB numpy array from Gradio.
    """
    if image is None:
        raise gr.Error("Please upload a Bangla handwritten page image to run OCR.")

    try:
        request_dir = _create_request_dir(None)
        logger.info("gradio request | dir={}", request_dir)

        success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            logger.error("gradio encode failed")
            raise gr.Error("Internal error while encoding the uploaded image.")
        original_bytes = buffer.tobytes()

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.info("gradio load_image | shape=({}, {})", bgr.shape[0], bgr.shape[1])

        segments, timings = run_handwritten_pipeline(bgr, request_dir=request_dir)
        logger.info("gradio pipeline done | segments={} timings={}", len(segments), timings)

        text = segments_to_text(segments)
        _save_artifacts(request_dir, original_bytes, text, segments, timings)
        logger.info("gradio outputs saved | dir={}", request_dir)

        # Optional detection overlay visualization
        overlay_rgb: Optional[np.ndarray] = None
        overlay_path = request_dir / "detection_boxes.png"
        if overlay_path.exists():
            try:
                overlay_bgr = cv2.imread(str(overlay_path))
                if overlay_bgr is not None:
                    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:  # pragma: no cover - logging only
                logger.warning("gradio overlay load failed | path={} error={}", overlay_path, e)

        json_path = request_dir / "ocr.json"
        json_path_str: Optional[str] = str(json_path) if json_path.exists() else None

        # Compute simple metadata for summary/status
        num_segments = len(segments)
        num_lines = len([ln for ln in text.splitlines() if ln.strip()]) if text else 0
        total_time = timings.get("pipeline_time") or (
            timings.get("total_segment_time", 0.0) + timings.get("word_rec_time", 0.0)
        )

        status_text = f"Completed detection and recognition in {total_time:.3f}s."
        meta_summary = (
            f"**Words:** {num_segments} &nbsp;|&nbsp; **Lines:** {num_lines} &nbsp;|&nbsp; "
            f"**Total time:** {total_time:.3f}s"
        )

        messages: List[str] = []
        if num_segments == 0:
            messages.append(
                "No text regions were detected. Check that the page contains Bangla handwriting, "
                "is reasonably sharp, and that the full page is visible."
            )
        h, w = bgr.shape[:2]
        if min(h, w) < 600:
            messages.append(
                f"Input resolution is relatively low ({w}×{h}). Higher-resolution scans or photos "
                "will generally improve OCR quality."
            )
        messages_text = "<br/>".join(messages) if messages else "No warnings."

        return text, overlay_rgb, timings, json_path_str, status_text, meta_summary, messages_text
    except gr.Error:
        # Pass through explicit Gradio errors as-is
        raise
    except Exception as e:  # pragma: no cover - top-level guard
        logger.exception("gradio _run_sensescan failed | error={}", e)
        raise gr.Error(
            "Unexpected error while running SenseScan. Please verify the image and try again."
        )


def build_interface() -> gr.Blocks:
    # Use default (light) theme for better contrast in demos.
    with gr.Blocks(title="SenseScan – Bangla Handwritten OCR") as demo:
        gr.Markdown(
            "## SenseScan – Bangla Handwritten Document OCR\n"
            "A focused demo of SenseScan's Bangla handwritten page OCR.\n\n"
            "**How to use:** Upload a single full-page Bangla handwritten image on the left, "
            "then click **Run SenseScan OCR** to view recognized text, visualizations, and timings.\n\n"
            "**Note:** This service currently supports only Bangla handwritten text."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Bangla handwritten page",
                    type="numpy",
                    image_mode="RGB",
                )
                gr.Examples(
                    examples=[
                        "test-data/hwrw_133_650_4.jpg",
                        "test-data/hwrw_265_1131_2.jpg",
                        "test-data/hwrw_1902_1285_5.jpg",
                        "test-data/hwrw_115_653_1.jpg",
                    ],
                    inputs=image_input,
                    label="Try with a sample handwritten page",
                )
                gr.Markdown(
                    "**Best results:**\n"
                    "- Use a single full page per image\n"
                    "- Keep the page reasonably flat and in focus\n"
                    "- Avoid strong shadows and extreme skew\n"
                    "- Bangla handwriting only (no printed/English text)"
                )
                run_button = gr.Button("Run SenseScan OCR", variant="primary")
                clear_button = gr.Button("Clear")

            with gr.Column(scale=2):
                with gr.Tab("Text"):
                    status_output = gr.Markdown(
                        value="Idle. Upload a page and click **Run SenseScan OCR**.",
                        elem_id="status-text",
                    )
                    meta_output = gr.Markdown(value="", elem_id="meta-summary")
                    text_output = gr.Textbox(
                        label="Recognized text",
                        lines=12,
                        show_copy_button=True,
                        interactive=False,
                    )
                with gr.Tab("Visualization"):
                    overlay_output = gr.Image(
                        label="Detected word regions (preview)",
                        interactive=False,
                    )
                with gr.Tab("Timings & JSON"):
                    timings_output = gr.JSON(label="Timing breakdown (seconds)")
                    json_download = gr.File(
                        label="Download full JSON result",
                        interactive=False,
                    )
                with gr.Tab("Messages"):
                    messages_output = gr.Markdown(
                        label="Warnings and informational messages",
                        value="No messages yet.",
                    )

        run_button.click(
            fn=lambda: "Running SenseScan OCR…",
            inputs=None,
            outputs=status_output,
        ).then(
            fn=_run_sensescan,
            inputs=[image_input],
            outputs=[
                text_output,
                overlay_output,
                timings_output,
                json_download,
                status_output,
                meta_output,
                messages_output,
            ],
            concurrency_limit=4,
            queue=True,
        )

        clear_button.click(
            fn=lambda: (
                None,
                "",
                None,
                "Idle. Upload a page and click **Run SenseScan OCR**.",
                "",
                "No messages yet.",
            ),
            inputs=None,
            outputs=[
                image_input,
                text_output,
                overlay_output,
                timings_output,
                status_output,
                meta_output,
                messages_output,
            ],
        )

    return demo


def main() -> None:
    demo = build_interface()

    port = int(os.environ.get("SENSESCAN_GRADIO_PORT", "8002"))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
        max_threads=4,
    )


if __name__ == "__main__":
    main()

