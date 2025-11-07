from types import SimpleNamespace
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from font_classifier.storia_ai import StoriaFontClassifier
from ocr_service.text_detector import TextDetector
from ocr_service.text_recognizer import TextRecognizer
from text_extract.ocr import run_ocr_on_crops
from utils.ocr_util import pil_to_bgr, bgr_to_pil, xyxy_cluster_to_xywh, ensure_clusters

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
    if "detectors" not in st.session_state:
        st.session_state.detectors = {}
    if "recognizers" not in st.session_state:
        st.session_state.recognizers = {}

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
        canvas_width = st.number_input("Canvas width (pixels)", 100, 900, 200, step=50)
        stroke_width = st.slider("Box border width", 1, 8, 3)

        use_ocr = ["tesseract", "paddle", "easyocr", "pororo"]
        recognizer_model = st.radio("Choose OCR model (for Recognition)", use_ocr)

        do_font = st.checkbox("Identify font (Storia-AI)", value=True)
        st.caption("Unchecked = deterministic stub OCR for MRE.")

        st.markdown("---")
        st.markdown("### Auto-Detection")

        detector_models = ["paddle", "easyocr", "pororo"]
        detector_model = st.radio("Choose Detector Model", detector_models)
        run_detector = st.button("Run Auto-Detection")

        if st.button("Clear All Boxes"):
            st.session_state.boxes = []

    return canvas_width, stroke_width, recognizer_model, do_font, detector_model, run_detector


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
    if trigger:
        with st.spinner(f"Running {model_name} detection..."):
            detector = get_detector_instance(model_name)
            detected_boxes, textlines_images, _ = detector.detect_text(img_bgr)

            st.session_state.text_lines = textlines_images
            st.session_state.boxes = detected_boxes
            st.success(f"Detected {len(detected_boxes)} boxes with {model_name}.")

def render_canvas(pil_img: Image.Image, canvas_width: int, display_height: int, stroke_width: int):
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

def canvas_to_boxes(canvas_result, W: int, H: int, canvas_width: int, display_height: int):
    boxes = []
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
                boxes.append([x, x + w, y, y + h])
    return boxes

# ---------------------------------------------------------------------------
# Visualization & Results
# ---------------------------------------------------------------------------

def draw_boxes_overlay(img_bgr: np.ndarray, boxes: List[List]) -> np.ndarray:
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    print(boxes)
    box_counter = 0
    for cluster in boxes:
        for c in cluster:
            box_counter += 1
            x1, x2, y1, y2 = map(int, c)
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 120, 255), 2)
                cv2.putText(
                    vis, f"{box_counter}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA
                )
    return vis

def render_selected_boxes(img_bgr: np.ndarray, boxes: List[List]) -> None:
    st.subheader("Selected boxes (mapped to original)")
    H, W = img_bgr.shape[:2]
    st.write(f"Image size: **{W}x{H}**, Boxes: **{len(boxes)}**")
    if boxes:
        vis = draw_boxes_overlay(img_bgr, boxes)
        st.image(bgr_to_pil(vis), caption="Overlay on original resolution", use_container_width=True)
    else:
        st.warning("No boxes drawn yet.")

def get_font_classifier() -> StoriaFontClassifier:
    if st.session_state["storia_font"] is None:
        st.session_state["storia_font"] = StoriaFontClassifier(topk=3)
    return st.session_state["storia_font"]

def create_default_opt(call_num: int) -> SimpleNamespace:
    if call_num == 0:
        return SimpleNamespace(
            cuda=torch.cuda.is_available(),
            TextDetector_weights_dir="ocr_service/weights/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_size=20,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.0,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.5,
            width_ths=0.5,
            add_margin=0.1
        )
    elif call_num == 1:
        return SimpleNamespace(
            device="cuda" if torch.cuda.is_available() else "cpu",
            saved_model = "ocr_service/weights/text_recognizer/TPS-ResNet-BiLSTM-CTC.pth",
            batch_size=192, # NOTE: ???
            batch_max_length = 25,
            imgH = 32,
            imgW = 100,
            PAD = False,
            rgb = False,
            Transformation = "TPS",
            FeatureExtraction = "ResNet",
            SequenceModeling = "BiLSTM",
            Prediction = "CTC",
            num_fiducial = 20,
            input_channel = 1,
            output_channel = 512,
            hidden_size = 256,
        )

def get_detector_instance(model_name: str) -> TextDetector:
    if model_name not in st.session_state.detectors:
        with st.spinner(f"Loading {model_name} detector model..."):
            default_opt = create_default_opt(0)
            st.session_state.detectors[model_name] = TextDetector(
                model_type=model_name,
                opt=default_opt
            )
    return st.session_state.detectors[model_name]

def get_recognizer_instance(model_name: str) -> TextRecognizer:
    if model_name not in st.session_state.recognizers:
        with st.spinner(f"Loading {model_name} recognizer model..."):
            default_opt = create_default_opt(1)
            st.session_state.recognizers[model_name] = TextRecognizer(
                model_type=model_name,
                opt=default_opt
            )
    return st.session_state.recognizers[model_name]

def render_ocr_results(img_bgr: np.ndarray, boxes: List[List], recognizer_model: str, do_font: bool) -> None:
    st.subheader("OCR results")

    if not boxes:
        st.info("Draw one or more rectangles to run OCR.")
        return

    H, W = img_bgr.shape[:2]
    clusters = ensure_clusters(boxes)

    for ci, cluster in enumerate(clusters, 1):
        # 클러스터 전체 크롭(컨텍스트용)
        xs1 = [int(b[0]) for b in cluster]
        xs2 = [int(b[1]) for b in cluster]
        ys1 = [int(b[2]) for b in cluster]
        ys2 = [int(b[3]) for b in cluster]
        cx1, cy1 = max(0, min(xs1)), max(0, min(ys1))
        cx2, cy2 = min(W, max(xs2)), min(H, max(ys2))
        if cx2 <= cx1 or cy2 <= cy1:
            st.warning(f"[Cluster {ci}] invalid union bbox skipped.")
            continue

        cluster_crop = img_bgr[cy1:cy2, cx1:cx2]
        st.markdown(f"### 🧩 Cluster {ci}")
        st.image(bgr_to_pil(cluster_crop), caption=f"Cluster {ci} crop", use_container_width=True)

        # 이 클러스터 안의 박스들을 xywh로 변환 후 OCR
        xywh_list = xyxy_cluster_to_xywh(cluster)
        # 경계 클램프
        xywh_list = [[max(0, min(x, W - 1)), max(0, min(y, H - 1)),
                      max(1, min(w, W - max(0, min(x, W - 1)))),
                      max(1, min(h, H - max(0, min(y, H - 1))))] for x, y, w, h in xywh_list]

        # 클러스터 단위로 OCR 실행
        results = run_ocr_on_crops(img_bgr, xywh_list, recognizer_model)

        # 검출 결과 "두 줄" 출력: ① 좌표, ② 텍스트
        for i, (b_xywh, txt) in enumerate(results, 1):
            x, y, w, h = map(int, b_xywh)

            # (1) 좌표 줄
            # st.markdown(f"**#{ci}-{i}** Box(x={x}, y={y}, w={w}, h={h})")
            # (2) 텍스트 줄
            st.code(txt or "[no text]")
            # 박스별 크롭 이미지도 같이 표시
            crop = img_bgr[y:y + h, x:x + w]
            # st.image(bgr_to_pil(crop), caption=f"Crop #{ci}-{i}", use_container_width=True)

            if do_font:
                clf = get_font_classifier()
                preds = clf.predict_topk(crop, topk=3)
                if not clf.is_ready and clf.error:
                    st.warning(f"Font model fallback (stub). Reason: {clf.error}")
                st.markdown("**Top fonts:**")
                for p in preds:
                    st.write(f"- `{p.filename}` → **{p.family}**  (p={p.score:.3f})  | [Google Fonts]({p.google_url})")
    ''' 
    if not boxes:
        st.info("Draw one or more rectangles to run OCR.")
        return

    text_boxes = st.session_state.text_lines
    recognizer = get_recognizer_instance(recognizer_model)
    recognized_texts = recognizer.extract_text(text_boxes)

    st.session_state.recognized_texts = recognized_texts

    st.markdown("### 🔍 Recognized Texts")
    for cluster_idx, text_list in enumerate(recognized_texts):
        joined_text = " ".join(text_list).strip()
        st.markdown(f"**Cluster {cluster_idx + 1}:**")
        st.code(joined_text if joined_text else "[no text detected]")
        if do_font:
            clf = get_font_classifier()
            preds = clf.predict_topk(text_boxes[cluster_idx], topk=3)
            if not clf.is_ready and clf.error:
                st.warning(f"Font model fallback (stub). Reason: {clf.error}")
            st.markdown("**Top fonts:**")
            for p in preds:
                st.write(f"- `{p.filename}` → **{p.family}**  (p={p.score:.3f})  | [Google Fonts]({p.google_url})")
   '''

def main() -> None:
    import logging
    logging.basicConfig(level=logging.DEBUG)

    setup_page()
    init_session_state()

    canvas_width, stroke_width, recognizer_model, do_font, detector_model, run_detector = sidebar_controls()
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
        render_ocr_results(img_bgr, boxes, recognizer_model, do_font)

if __name__ == "__main__":
    main()
