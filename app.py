from typing import List

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

if "boxes" not in st.session_state:
    st.session_state.boxes = []

def draw_boxes_overlay(img_bgr: np.ndarray, boxes: List[Box]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, b in enumerate(boxes, 1):
        cv2.rectangle(vis, (b.x, b.y), (b.x+b.w, b.y+b.h), (0, 120, 255), 2)
        cv2.putText(vis, f"{i}", (b.x, max(0, b.y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,120,255), 2, cv2.LINE_AA)
    return vis

st.set_page_config(page_title="Manual ROI OCR", layout="wide")
st.title("Manual Text Region Picker (Draw rectangles)")

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
    detector = st.button("Run Auto-Detection")
    # 박스 초기화 버튼 추가
    if st.button("Clear All Boxes"):
        st.session_state.boxes = []

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])

if uploaded is None:
    st.info("Upload an image to start, then draw one or more rectangles.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(pil)
H, W = img_bgr.shape[:2]
aspect = H/W
display_height = int(canvas_width * aspect)

if uploaded is not None:
    if detector:
        with st.spinner(f"Running {detector_model} detection..."):
            detected_boxes = run_detector_on_image(img_bgr, detector_model)
            st.session_state.boxes = detected_boxes
            st.success(f"Detected {len(detected_boxes)} boxes.")
    else:
        st.warning("Please upload an image first.")

st.subheader("Draw rectangles on the image")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#ff8800",
    background_image=pil.resize((canvas_width, display_height)),
    update_streamlit=True,
    height=display_height,
    width=canvas_width,
    drawing_mode="rect",
    key="canvas",
)

boxes: List[Box] = []
if canvas_result.json_data is not None:
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

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Selected boxes (mapped to original)")
    st.write(f"Image size: **{W}x{H}**, Boxes: **{len(boxes)}**")
    if boxes:
        vis = draw_boxes_overlay(img_bgr, boxes)
        st.image(bgr_to_pil(vis), caption="Overlay on original resolution", use_container_width=True)
    else:
        st.warning("No boxes drawn yet.")

with col2:
    st.subheader("OCR results")
    if boxes:
        results = run_ocr_on_crops(img_bgr, boxes, ocr_model)
        for i, (b, txt) in enumerate(results, 1):
            st.markdown(f"**#{i}** Box(x={b.x}, y={b.y}, w={b.w}, h={b.h})")
            st.code(txt or "[no text]")
            crop = img_bgr[b.y:b.y+b.h, b.x:b.x+b.w]
            st.image(bgr_to_pil(crop), caption=f"Crop #{i}", use_container_width=True)

            if do_font:
                if "storia_font" not in st.session_state:
                    st.session_state["storia_font"] = StoriaFontClassifier(topk=3)
                clf = st.session_state["storia_font"]
                preds = clf.predict_topk(crop, topk=3)

                if not clf.is_ready and clf.error:
                    st.warning(f"Font model fallback (stub). Reason: {clf.error}")
                st.markdown("**Top fonts:**")
                for p in preds:
                    st.write(f"- `{p.filename}` → **{p.family}**  (p={p.score:.3f})  | [Google Fonts]({p.google_url})")

    else:
        st.info("Draw one or more rectangles to run OCR.")
