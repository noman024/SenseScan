## SenseScan

SenseScan is a lightweight web UI for an OCR (Optical Character Recognition) backend service.
It is built with Gradio and talks to a FastAPI server that exposes an `/ml/ocr` endpoint.

The UI lets you:

- Upload an **image** or **PDF**
- Send it to the remote OCR service
- View the recognized content as **Markdown**, a **visualized layout image**, and raw **JSON**
- Automatically save the input and outputs next to the original file on disk

---

## Project structure

- `dev_ocr_ui.py`: Gradio-based UI for interacting with the OCR backend.
- `requirements.txt`: Python dependencies needed to run the UI.

You are expected to run a separate OCR FastAPI server that implements the `/ml/ocr` endpoint.

---

## Prerequisites

- **Python**: 3.10+ recommended
- **pip**: latest version

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <YOUR_REPO_URL> SenseScan
   cd SenseScan
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

The UI needs to know where your **OCR FastAPI dev server** is running.

- **Environment variable**: `OCR_DEV_BASE_URL`
- **Default**: `http://100.116.65.82:9192`

Examples:

```bash
export OCR_DEV_BASE_URL="http://localhost:9192"
python dev_ocr_ui.py
```

Or in a single command:

```bash
OCR_DEV_BASE_URL="http://localhost:9192" python dev_ocr_ui.py
```

The UI will call:

- `GET/POST`: `${OCR_DEV_BASE_URL}/ml/ocr`

and expects a **ZIP file** response containing:

- A Markdown file (`*.md`)
- A JSON file (`*.json`)
- An optional visualization image (`*.png` / `*.jpg`)

---

## Running the UI

With your OCR backend running and dependencies installed:

```bash
python dev_ocr_ui.py
```

By default, the app:

- Listens on `0.0.0.0:7860`
- Enables a Gradio queue for concurrent requests
- May expose a public `share` URL (via Gradio) for quick testing

Once started, open the printed URL in your browser.

---

## Using the app

1. Open the UI in your browser.
2. **Upload an image** (preferred) or a **PDF**.
3. Click **“Run OCR”**.
4. Inspect the results in the tabs:
   - **Markdown**: recognized text content, with inline images rendered.
   - **Visualized Image**: layout visualization returned by the backend.
   - **JSON**: raw structured output from the OCR service.

For each input, the app saves everything under a **`data/`** directory:

- Creates `data/<name>/` (e.g. `data/mydoc/`)
- Saves there:
  - a copy of the input
  - `<name>.md`
  - `<name>.json`
  - `<name>_image.png` (if available)
  - any inline images extracted from base64 data URIs

---

## Development notes

- The UI is defined in `build_ui()` inside `dev_ocr_ui.py`.
- The OCR call logic lives in `call_ocr_endpoint()` and expects a ZIP response structure as described above.
- Image artifacts created from inline base64 data URIs are stored as `inline_*.png`/`.jpg`/`.gif` in `data/<input_name>/`.

If you change the backend contract (e.g. different endpoint or response format), update `call_ocr_endpoint()` accordingly.

---

## Troubleshooting

- **No file uploaded**: Ensure you upload either an image or a PDF before clicking **Run OCR**.
- **Connection / timeout errors**:
  - Check that your OCR FastAPI server is running.
  - Verify `OCR_DEV_BASE_URL` is set correctly and reachable from this machine.
- **Server returned HTTP 4xx/5xx**:
  - Inspect the error text shown in the Markdown/JSON tab.
  - Check your backend logs for more details.

---

## License

Add your chosen license here (for example, MIT, Apache-2.0, or proprietary).

