import os
import io
import zipfile
import json
import mimetypes
import tempfile
import re
import base64
import hashlib
from typing import Tuple, Optional

import requests
from PIL import Image
import gradio as gr


def get_ocr_base_url() -> str:
    """
    Base URL for the dev OCR FastAPI server.
    Example: http://100.116.65.82:9192
    """
    return os.environ.get("OCR_DEV_BASE_URL", "http://100.116.65.82:9192")


# Match markdown images whose URL is a data:image base64 URI:
# ![](data:image/png;base64,...)
MD_DATA_IMG_PATTERN = re.compile(
    r"!\[[^\]]*\]\((data:image/(?:png|jpe?g|gif)[^)]*)\)"
)

# Match bare data:image base64 URIs (across newlines if present)
DATA_URI_PATTERN = re.compile(
    r"(data:image/(?:png|jpe?g|gif)[^;]*;base64,[A-Za-z0-9+/=\r\n]+)"
)

# Match HTML <img> tags that use a data:image base64 src
IMG_TAG_PATTERN = re.compile(
    r"<img[^>]+src=[\"'](data:image/[^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)

# Match nested form produced by earlier transforms:
# ![](![image](data:image/...))
NESTED_MARKDOWN_IMG_PATTERN = re.compile(
    r"!\[\]\(!\[image\]\((data:image/[^)]+)\)\)"
)


def _data_uri_to_file(uri: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Decode a data:image/...;base64,... URI and save it as an image file
    under the given output directory (or CWD). Returns a path relative to
    the current working directory, or None on failure.
    """
    uri = uri.replace("\n", "").replace("\r", "")
    if "," not in uri:
        return None
    header, b64_data = uri.split(",", 1)
    # Infer extension from mime-type
    ext = "png"
    if "jpeg" in header or "jpg" in header:
        ext = "jpg"
    elif "gif" in header:
        ext = "gif"

    try:
        binary = base64.b64decode(b64_data, validate=False)
    except Exception:
        return None

    try:
        digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]
        filename = f"inline_{digest}.{ext}"
        base_dir = output_dir or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, filename)
        with open(path, "wb") as f:
            f.write(binary)
        # Return a path relative to the Gradio app's working directory so it
        # can be referenced via the `file/` prefix from the frontend.
        rel_path = os.path.relpath(path, os.getcwd())
        return rel_path
    except Exception:
        return None


def _ensure_base64_markdown_images(
    markdown_str: str, output_dir: Optional[str] = None
) -> str:
    """
    Convert any data:image base64 usages (markdown or HTML) into concrete
    image files on disk, and rewrite them to HTML <img> tags that reference
    those files via the `file/` prefix so that Gradio's Markdown renderer
    can display the images inline.
    """

    # 0) Fix nested `![](![image](data:...))` into a clean `![](data:...)`
    text = NESTED_MARKDOWN_IMG_PATTERN.sub(
        lambda m: f"![]({m.group(1)})", markdown_str
    )

    # 1) Handle markdown image: ![](data:image/...)
    def repl_md_img(match: re.Match) -> str:
        uri = match.group(1)
        rel_path = _data_uri_to_file(uri, output_dir=output_dir)
        if not rel_path:
            return match.group(0)
        # Gradio serves local files via the /gradio_api/file=<path> endpoint.
        return f'<img src="/gradio_api/file={rel_path}" />'

    text = MD_DATA_IMG_PATTERN.sub(repl_md_img, text)

    # 2) Handle HTML <img src="data:image/..."> tags
    def repl_img_tag(match: re.Match) -> str:
        uri = match.group(1)
        rel_path = _data_uri_to_file(uri, output_dir=output_dir)
        if not rel_path:
            return match.group(0)
        return f'<img src="/gradio_api/file={rel_path}" />'

    text = IMG_TAG_PATTERN.sub(repl_img_tag, text)

    # 3) Handle any remaining bare data:image... occurrences
    def repl_data_uri(match: re.Match) -> str:
        uri = match.group(1)
        rel_path = _data_uri_to_file(uri, output_dir=output_dir)
        if not rel_path:
            return match.group(0)
        return f'<img src="/gradio_api/file={rel_path}" />'

    text = DATA_URI_PATTERN.sub(repl_data_uri, text)
    return text


def call_ocr_endpoint(
    file_obj,
    prompt_mode: str = "prompt_layout_all_en",
) -> Tuple[str, Optional[Image.Image], str]:
    """
    Call the remote /ml/ocr endpoint and return (markdown, image, json_text).
    """

    if file_obj is None:
        return "No file uploaded.", None, "No JSON."

    base_url = get_ocr_base_url().rstrip("/")
    url = f"{base_url}/ml/ocr"

    # Gradio can return a dict, a filepath string, or a file-like object
    # depending on version and component type. Normalize to (file_path, name).
    # Try to robustly extract the file path and name.
    file_path = None
    file_name = "upload"

    if isinstance(file_obj, dict) and "name" in file_obj:
        file_path = file_obj["name"]
        file_name = os.path.basename(file_path)
    elif isinstance(file_obj, str):
        file_path = file_obj
        file_name = os.path.basename(file_path)
    elif hasattr(file_obj, "name"):
        # tempfile or file-like
        file_path = file_obj.name
        file_name = os.path.basename(file_path)
    else:
        raise RuntimeError("Unsupported file object type from Gradio.")

    if not os.path.exists(file_path):
        return "Uploaded file path not found on disk.", None, "No JSON."

    # Directory for this input's artifacts (under data/, based on input file name)
    base_name, input_ext = os.path.splitext(file_name)
    output_dir = os.path.join(os.getcwd(), "data", base_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        # Fallback to data/ if we cannot create the named subdirectory
        output_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(output_dir, exist_ok=True)

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as f:
        files = {"file": (file_name, f, mime_type)}
        data = {"prompt_mode": prompt_mode}

        try:
            resp = requests.post(url, files=files, data=data, timeout=600)
        except Exception as e:
            return f"Request error: {e}", None, "Request failed."

    if resp.status_code != 200:
        return (
            f"Server returned HTTP {resp.status_code}: {resp.text[:500]}",
            None,
            f"Error:\n{resp.text[:2000]}",
        )

    # Response is a ZIP containing md, json, images, etc.
    zip_bytes = io.BytesIO(resp.content)
    try:
        zf = zipfile.ZipFile(zip_bytes)
    except Exception as e:
        return f"Failed to parse ZIP response: {e}", None, "Invalid ZIP from server."

    namelist = zf.namelist()

    # Helper to pick a file by extension, optionally preferring non-*nohf* variants for md.
    def pick_first(exts, prefer_without_nohf=False):
        candidates = [n for n in namelist if any(n.lower().endswith(ext) for ext in exts)]
        if not candidates:
            return None
        if prefer_without_nohf:
            non_nohf = [c for c in candidates if "_nohf" not in c.lower()]
            if non_nohf:
                return sorted(non_nohf)[0]
        return sorted(candidates)[0]

    md_file = pick_first([".md"], prefer_without_nohf=True)
    json_file = pick_first([".json"])
    img_file = pick_first([".jpg", ".jpeg", ".png"])

    # Markdown content
    markdown_str = "No markdown file (.md) found in response."
    if md_file:
        try:
            md_bytes = zf.read(md_file)
            markdown_str = md_bytes.decode("utf-8", errors="replace")
            markdown_str = _ensure_base64_markdown_images(
                markdown_str, output_dir=output_dir
            )
        except Exception as e:
            markdown_str = f"Failed to read markdown file {md_file}: {e}"

    # JSON content
    json_display = "No JSON file (.json) found in response."
    if json_file:
        try:
            json_bytes = zf.read(json_file)
            json_str = json_bytes.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(json_str)
                json_display = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # Show raw text if not valid JSON
                json_display = json_str
        except Exception as e:
            json_display = f"Failed to read JSON file {json_file}: {e}"

    # Image content
    pil_image = None
    if img_file:
        try:
            img_bytes = zf.read(img_file)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            pil_image = None

    # Save input and outputs into a dedicated directory named after the input.
    try:
        # Save a copy of the input file
        try:
            input_dest = os.path.join(output_dir, f"{base_name}_input{input_ext}")
            if os.path.abspath(file_path) != os.path.abspath(input_dest):
                with open(file_path, "rb") as src, open(input_dest, "wb") as dst:
                    dst.write(src.read())
        except Exception:
            pass

        # Save markdown
        try:
            md_dest = os.path.join(output_dir, f"{base_name}.md")
            with open(md_dest, "w", encoding="utf-8") as f_md:
                f_md.write(markdown_str)
        except Exception:
            pass

        # Save JSON
        try:
            json_dest = os.path.join(output_dir, f"{base_name}.json")
            with open(json_dest, "w", encoding="utf-8") as f_js:
                f_js.write(json_display)
        except Exception:
            pass

        # Save visualized image (if available)
        if pil_image is not None:
            try:
                img_dest = os.path.join(output_dir, f"{base_name}_image.png")
                pil_image.save(img_dest, format="PNG")
            except Exception:
                pass
    except Exception:
        # Never let file-saving issues break the UI
        pass

    return markdown_str, pil_image, json_display


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SenseScan") as demo:
        gr.Markdown("# SenseScan")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload image",
                    sources=["upload"],
                    type="filepath",
                    height=256,
                )
                file_input = gr.File(
                    label="Upload PDF (optional)",
                    file_types=[".pdf"],
                    type="filepath",
                )
                run_btn = gr.Button("Run OCR", variant="primary")
                status = gr.Markdown(
                    value="**Status:** Idle.",
                    visible=True,
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Markdown"):
                        md_out = gr.Markdown(
                            value="Markdown output will appear here."
                        )
                        inline_gallery = gr.Gallery(
                            label="Inline images",
                            show_label=False,
                            columns=[4],
                            height=200,
                        )
                    with gr.Tab("Visualized Image"):
                        img_out = gr.Image(
                            label="Layout / visualization",
                            type="pil",
                            interactive=False,
                        )
                    with gr.Tab("JSON"):
                        json_out = gr.Code(
                            label="JSON output",
                            language="json",
                            value="// JSON output will appear here.",
                        )

        def run_and_update(image_file, pdf_file):
            # Initial loading state
            yield (
                "Running OCR, please wait...",
                [],
                None,
                "// Running OCR, please wait...",
                "**Status:** Running OCR...",
            )

            # Prefer image input if provided; otherwise fall back to PDF file.
            chosen_file = image_file or pdf_file

            md, img, js = call_ocr_endpoint(chosen_file)

            # Collect any inline images saved for this input.
            inline_paths = []
            try:
                if chosen_file is not None:
                    file_path = None
                    if isinstance(chosen_file, dict) and "name" in chosen_file:
                        file_path = chosen_file["name"]
                    elif isinstance(chosen_file, str):
                        file_path = chosen_file
                    elif hasattr(chosen_file, "name"):
                        file_path = chosen_file.name

                    if file_path and os.path.exists(file_path):
                        base_name, _ = os.path.splitext(os.path.basename(file_path))
                        out_dir = os.path.join(os.getcwd(), "data", base_name)
                        if os.path.isdir(out_dir):
                            for name in sorted(os.listdir(out_dir)):
                                lower = name.lower()
                                if lower.startswith("inline_") and lower.endswith(
                                    (".png", ".jpg", ".jpeg", ".gif")
                                ):
                                    inline_paths.append(os.path.join(out_dir, name))
            except Exception:
                inline_paths = []

            if chosen_file is None:
                status_text = "**Status:** No file uploaded."
            elif isinstance(md, str) and (
                md.startswith("Request error")
                or md.startswith("Server returned HTTP")
                or md.startswith("Failed")
                or md.startswith("Uploaded file path not found")
            ):
                status_text = "**Status:** Error while processing."
            else:
                status_text = "**Status:** Completed."

            yield (
                md,
                inline_paths,
                img,
                js,
                status_text,
            )

        run_btn.click(
            fn=run_and_update,
            inputs=[image_input, file_input],
            outputs=[md_out, inline_gallery, img_out, json_out, status],
            show_progress="minimal",
            concurrency_limit=16,
        )

    return demo


def main():
    demo = build_ui()
    # Enable a request queue so multiple users can use the app concurrently.
    # Configure the queue with a higher concurrency limit for this app.
    demo.queue(default_concurrency_limit=16, max_size=128)
    # share=True gives you a public URL you can share
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        allowed_paths=["."],
    )


if __name__ == "__main__":
    main()