from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from font_classifier.storia_ai import StoriaFontClassifier
from text_extract.data import Box
from text_extract.ocr import run_ocr_on_crops
from text_extract.detector import run_detector_on_image
from utils.ocr_util import pil_to_bgr, bgr_to_pil

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    if "boxes" not in st.session_state:
        st.session_state.boxes = []
    if "storia_font" not in st.session_state:
        st.session_state["storia_font"] = None

# ---------------------------------------------------------------------------
# UI: Page & Sidebar
# ---------------------------------------------------------------------------
def setup_page() -> None:
    st.set_page_config(page_title="OCR", layout="wide")
    st.title("OCR 테스트 공간")

def sidebar_controls() -> Tuple[int, int, str, bool, str, bool]:
    """
    사이드 바. 유저 선택 결과 반환

    Returns:
        canvas_width: int
        stroke_width: int
        ocr_model: str
        do_font: bool
        detector_model: str
        run_detector: bool
    """
    with st.sidebar:
        st.markdown("### Options")
        canvas_width = st.number_input("Canvas width (pixels)", 400, 1600, 900, step=50)
        stroke_width = st.slider("Box border width", 1, 8, 3)

        use_ocr = ["tesseract", "paddle", "easyocr", "pororo"]
        ocr_model = st.radio("Choose OCR model (for Recognition)", use_ocr)

        do_font = st.checkbox("Identify font (Storia-AI)", value=True)
        st.caption("Unchecked = deterministic stub OCR for MRE.")

        st.markdown("---")
        st.markdown("### Auto-Detection")

        detector_models = ["paddle", "easyocr", "pororo"]
        detector_model = st.radio("Choose Detector Model", detector_models)
        run_detector = st.button("Run Auto-Detection")

        if st.button("Clear All Boxes"):
            st.session_state.boxes = []

    return canvas_width, stroke_width, ocr_model, do_font, detector_model, run_detector


def load_uploaded_image() -> Tuple[Image.Image, np.ndarray]:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to start, then draw one or more rectangles.")
        st.stop()

    pil = Image.open(uploaded).convert("RGB")
    img_bgr = pil_to_bgr(pil)
    return pil, img_bgr

def compute_display_dims(img_bgr: np.ndarray, canvas_width: int) -> Tuple[int, int, int, int]:
    H, W = img_bgr.shape[:2]
    aspect = H / W
    display_height = int(canvas_width * aspect)
    return H, W, display_height, canvas_width

# ---------------------------------------------------------------------------
# Detection & Canvas
# ---------------------------------------------------------------------------
def maybe_run_auto_detection(img_bgr: np.ndarray, model_name: str, trigger: bool) -> None:
    """
    Optionally run detector on the full image. Keeps behavior identical to the original:
    - Save results to st.session_state.boxes
    - Show a toast message
    """
    if trigger:
        with st.spinner(f"Running {model_name} detection..."):
            detected_boxes = run_detector_on_image(img_bgr, model_name)
            st.session_state.boxes = detected_boxes
            st.success(f"Detected {len(detected_boxes)} boxes.")
    else:
        # Matches original code path (no-op for detector until user presses the button)
        pass

def render_canvas(pil_img: Image.Image, canvas_width: int, display_height: int, stroke_width: int):
    """Render drawable canvas and return the canvas result."""
    st.subheader("Draw rectangles on the image")
    return st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#ff8800",
        background_image=pil_img.resize((canvas_width, display_height)),
        update_streamlit=True,
        height=display_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
    )

def canvas_to_boxes(
    canvas_result,
    W: int,
    H: int,
    canvas_width: int,
    display_height: int
) -> List[Box]:
    """
    Convert canvas rectangles (in display coords) to original image coords.
    Behavior stays the same as original app—only rectangles are captured.
    """
    boxes: List[Box] = []
    if canvas_result and canvas_result.json_data is not None:
        for obj in canvas_result.json_data.get("objects", []):
            if obj.get("type") != "rect":
                continue
            cx, cy = float(obj.get("left", 0)), float(obj.get("top", 0))
            cw, ch = float(obj.get("width", 0)), float(obj.get("height", 0))
            sx, sy = W / canvas_width, H / display_height
            x, y = int(round(cx * sx)), int(round(cy * sy))
            w, h = int(round(cw * sx)), int(round(ch * sy))
            if w > 0 and h > 0:
                boxes.append(Box(x, y, w, h))
    return boxes

# ---------------------------------------------------------------------------
# Visualization & Results
# ---------------------------------------------------------------------------
def draw_boxes_overlay(img_bgr: np.ndarray, boxes: List[Box]) -> np.ndarray:
    """Overlay rectangular boxes and indices on the image."""
    vis = img_bgr.copy()
    for i, b in enumerate(boxes, 1):
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 120, 255), 2)
        cv2.putText(
            vis, f"{i}", (b.x, max(0, b.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA
        )
    return vis

def render_selected_boxes(img_bgr: np.ndarray, boxes: List[Box]) -> None:
    """Left column: show overlay preview."""
    st.subheader("Selected boxes (mapped to original)")
    H, W = img_bgr.shape[:2]
    st.write(f"Image size: **{W}x{H}**, Boxes: **{len(boxes)}**")
    if boxes:
        vis = draw_boxes_overlay(img_bgr, boxes)
        st.image(bgr_to_pil(vis), caption="Overlay on original resolution", use_container_width=True)
    else:
        st.warning("No boxes drawn yet.")

def get_font_classifier() -> StoriaFontClassifier:
    """Lazy-create StoriaFontClassifier and cache in session."""
    if st.session_state["storia_font"] is None:
        st.session_state["storia_font"] = StoriaFontClassifier(topk=3)
    return st.session_state["storia_font"]

def render_ocr_results(img_bgr: np.ndarray, boxes: List[Box], ocr_model: str, do_font: bool) -> None:
    """Right column: OCR text & optional font predictions for each crop."""
    st.subheader("OCR results")
    if not boxes:
        st.info("Draw one or more rectangles to run OCR.")
        return

    results = run_ocr_on_crops(img_bgr, boxes, ocr_model)
    for i, (b, txt) in enumerate(results, 1):
        st.markdown(f"**#{i}** Box(x={b.x}, y={b.y}, w={b.w}, h={b.h})")
        st.code(txt or "[no text]")

        crop = img_bgr[b.y:b.y + b.h, b.x:b.x + b.w]
        st.image(bgr_to_pil(crop), caption=f"Crop #{i}", use_container_width=True)

        if do_font:
            clf = get_font_classifier()
            preds = clf.predict_topk(crop, topk=3)
            if not clf.is_ready and clf.error:
                st.warning(f"Font model fallback (stub). Reason: {clf.error}")
            st.markdown("**Top fonts:**")
            for p in preds:
                st.write(f"- `{p.filename}` → **{p.family}**  (p={p.score:.3f})  | [Google Fonts]({p.google_url})")


def main() -> None:
    setup_page()
    init_session_state()

    canvas_width, stroke_width, ocr_model, do_font, detector_model, run_detector = sidebar_controls()
    pil_img, img_bgr = load_uploaded_image()
    H, W, display_height, _ = compute_display_dims(img_bgr, canvas_width)

    maybe_run_auto_detection(img_bgr, detector_model, run_detector)
    canvas_result = render_canvas(pil_img, canvas_width, display_height, stroke_width)
    manual_boxes = canvas_to_boxes(canvas_result, W, H, canvas_width, display_height)

    if manual_boxes:
        st.session_state.boxes = manual_boxes

    boxes = st.session_state.boxes
    col1, col2 = st.columns([1, 1])
    with col1:
        render_selected_boxes(img_bgr, boxes)
    with col2:
        render_ocr_results(img_bgr, boxes, ocr_model, do_font)

if __name__ == "__main__":
    main()
