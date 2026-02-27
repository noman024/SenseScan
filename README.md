## SenseScan – Bangla Handwritten Document OCR

- **Author:** MD Mutasim Billah Noman
- **Updated:** 2/27/2026

SenseScan is a focused **Bangla handwritten document OCR** service. It exposes a small,
production-friendly FastAPI app that:

- **Accepts** a full-page handwritten image (currently tuned for Bangla),
- **Runs** EAST-based text detection + CRNN word recognition (quantization-aware),
- **Returns** either:
  - plain-text OCR output via the `/plugin` endpoint, or
  - a structured JSON response with layout and simple timing info.

The core application lives entirely in the `sensescan` package.

---

## 1. Requirements

- **Python**: 3.10 (recommended; matches current virtualenv)
- **Hardware**:
  - CPU-only works, but **CUDA GPU(s)** are strongly recommended.
  - If multiple GPUs are available, the models automatically use `DataParallel`.
- **Disk space**:
  - Several GB for PyTorch + CUDA wheels.
  - Additional space for model checkpoints (not included in this repository).

---

## 2. Project layout

Key files and directories:

- `sensescan/`
  - `__init__.py` – package marker.
  - `config.py` – device / GPU and model-path configuration.
  - `detection.py` – EAST-based word segmentation (PyTorch + OpenCV + LANMS).
  - `recognition.py` – CRNN-based handwritten word recognizer (QAT-compatible).
  - `pipeline.py` – high-level handwritten OCR pipeline and text reconstruction.
  - `api.py` – FastAPI application exposing health, plain-text, and JSON endpoints.
- `requirements.txt`
  - Runtime dependencies for FastAPI + PyTorch + supporting libraries.
- `models/`
  - **Ignored by git** (see `.gitignore`).
  - You must place your EAST and CRNN checkpoints here (see below).

---

## 3. Setup

From the project root (e.g. `/home/noman/SenseScan`):

```bash
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a suitable CUDA-enabled PyTorch installed in the virtualenv,
you may comment out the `torch`/`torchvision`/`torchaudio` lines in
`requirements.txt` and manage them separately to save disk space.

---

## 4. Model files

SenseScan expects **two** trained PyTorch checkpoints (not included in this repo).
The canonical SenseScan filenames are:

- **EAST handwritten segmentation model**: `SENSESCAN_HW_EAST_MODEL.pth`
- **Handwritten CRNN word model**: `SENSESCAN_HW_WORD_MODEL.pth`

### 4.1. Default locations

By default, the app will look for:

```text
<project-root>/models/HW/SENSESCAN_HW_EAST_MODEL.pth
<project-root>/models/HW/SENSESCAN_HW_WORD_MODEL.pth
```

These paths are the current SenseScan defaults.

### 4.2. Recommended SenseScan-specific environment variables

You can override the default model paths using **SenseScan**-prefixed environment
variables:

```bash
export SENSESCAN_HW_EAST_MODEL_PATH=/absolute/path/to/SENSESCAN_HW_EAST_MODEL.pth
export SENSESCAN_HW_WORD_MODEL_PATH=/absolute/path/to/SENSESCAN_HW_WORD_MODEL.pth
```

---

## 5. Running the server

From the project root:

```bash
source venv/bin/activate

# Optional: restrict which GPUs to use
# export CUDA_VISIBLE_DEVICES=0,1

uvicorn sensescan.api:app --host 0.0.0.0 --port 8001
```

On startup, you should see log lines similar to:

```text
SenseScan using device: cuda:0 with DataParallel over 2 GPUs
SenseScan EAST model path: /home/noman/SenseScan/models/HW/SENSESCAN_HW_EAST_MODEL.pth
SenseScan handwritten word model path: /home/noman/SenseScan/models/HW/SENSESCAN_HW_WORD_MODEL.pth
```

---

## 6. API overview

SenseScan exposes a minimal, opinionated API surface:

- **Health**
  - `GET /health`
- **Plain-text OCR**
  - `POST /plugin`
- **Structured JSON OCR (recommended for new integrations)**
  - `POST /v1/ocr/handwritten`

All OCR endpoints are designed specifically for **single-page Bangla handwritten document
images**. Other languages/scripts are not supported.

---

## 7. Plain-text endpoint (`/plugin`)

### 7.1. Request

- **Method**: `POST`
- **Path**: `/plugin`
- **Content type**: `multipart/form-data`
- **Fields**:
  - `image`: uploaded image file (handwritten page)

### 7.2. Example with `curl`

```bash
curl -X POST \
  -F "image=@test-data/hwrw_1902_1285_5.jpg" \
  http://localhost:8001/plugin
```

### 7.3. Response

- **HTTP 200** on success.
- **Body**: plain-text OCR output (multiple lines), for example:

```text
ম ম
...
নূর মোহাম্মদ নূর
```

- No JSON wrapper is used; the body is **just the recognized text** in reading order.

---

## 8. JSON OCR endpoint (`/v1/ocr/handwritten`)

### 8.1. Request

- **Method**: `POST`
- **Path**: `/v1/ocr/handwritten`
- **Content type**: `multipart/form-data`
- **Fields**:
  - `image`: uploaded image file (handwritten page)
- **Query parameters**:
  - `include_segments` (bool, default `false`):
    - If `true`, include per-segment text and geometry.
  - `include_timings` (bool, default `false`):
    - If `true`, include a simple timing breakdown.

### 8.2. Example with `curl`

```bash
curl -X POST \
  -F "image=@test-data/hwrw_1902_1285_5.jpg" \
  "http://localhost:8001/v1/ocr/handwritten?include_segments=true&include_timings=true"
```

### 8.3. Response schema

```json
{
  "text": "full page text in reading order",
  "segments": [
    {
      "id": "uuid-1",
      "text": "word-or-token",
      "points": [
        { "x": 10.0, "y": 20.0 },
        { "x": 10.0, "y": 40.0 },
        { "x": 80.0, "y": 20.0 },
        { "x": 80.0, "y": 40.0 }
      ]
    }
    // ...
  ],
  "timings": {
    "total_segment_time": 0.1234,
    "word_rec_time": 0.4567,
    "pipeline_time": 0.7890
  }
}
```

- **`text`**:
  - Single string with `\n`-separated lines in reading order.
- **`segments`** (optional, only when `include_segments=true`):
  - One entry per detected word/segment.
  - `points` is a quadrilateral in **image coordinates**.
- **`timings`** (optional, only when `include_timings=true`):
  - Simple, high-level timing breakdown in seconds.

This JSON endpoint is the recommended way to integrate SenseScan into larger
systems (document viewers, labeling tools, indexing pipelines, etc.) where both
text and geometry are important.

---

## 9. Design notes: input and output shape

- **Input assumptions**:
  - One **full page** per request.
  - Dominantly handwritten Bangla; other scripts may work but are not tuned.
  - Reasonable resolution (EAST resizes while preserving aspect ratio).
- **Output philosophy**:
  - **`/plugin`** keeps the interface extremely simple for shell / CLI use:
    - Just text, nothing else.
  - **`/v1/ocr/handwritten`** is structured and explicit:
    - Separates **page text**, **segments**, and **timings**.
    - Uses UUIDs and explicit `(x, y)` points so clients can:
      - Render overlays,
      - Re-group into lines/paragraphs,
      - Associate text with downstream metadata.
- **Extensibility**:
  - Additional fields such as confidence scores, reading-order indices, or
    multi-page document IDs can be added as new optional fields in the JSON
    response without breaking existing clients.

---

## 10. Notes

- The CRNN word model is **quantization-aware trained (QAT)**; the architecture in
  `sensescan/recognition.py` is built to load such checkpoints correctly.
- Both the EAST detector and CRNN recognizer automatically use
  `torch.nn.DataParallel` when **more than one CUDA GPU** is visible.
- Occasional PyTorch quantization warnings (for example, around `_aminmax`
  deprecations) are expected and do not affect inference.

