import logging
from typing import List

import cv2
import numpy as np
import torch

from ocr_service.pororo.brainocr import Reader as PororoReader
from ocr_service.pororo.tasks.utils.download_utils import download_or_load
from text_extract.data import Box

logger = logging.getLogger(__name__)

_paddle_detector = None
_easy_detector = None
_pororo_detector = None


def run_detector_on_image(img_bgr: np.ndarray, detector_model: str) -> List[Box]:
    detector_functions = {
        "paddle": detect_paddle,
        "easyocr": detect_easyocr,
        "pororo": detect_pororo,
    }

    detect_func = detector_functions.get(detector_model)

    if detect_func is None:
        logger.error(f"Unknown OCR detector: {detector_model}. Supported: {list(detector_functions.keys())}")
        return []

    try:
        boxes = detect_func(img_bgr)
        logger.info(f"Detected {len(boxes)} text boxes using {detector_model}.")
        return boxes
    except Exception as e:
        logger.error(f"Failed to detect text regions with {detector_model}: {e}")
        return []


def get_paddle_detector():
    global _paddle_detector
    if _paddle_detector is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(lang='korean')
            logger.info("PaddleOCR detector loaded successfully.")
            return _paddle_detector
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR detector: {e}")
            _paddle_detector = None
    return _paddle_detector


def get_easy_detector():
    global _easy_detector
    if _easy_detector is None:
        try:
            from easyocr import Reader
            _easy_detector = Reader(lang_list=['ko', 'en'], gpu=True)
            logger.info("EasyOCR detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR detector: {e}")
            _easy_detector = None
    return _easy_detector


def get_pororo_detector():
    global _pororo_detector
    if _pororo_detector is None:
        try:
            logger.info("Loading Pororo OCR detector (using ocr_service)...")
            LANG = "ko"
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            det_model_path = download_or_load(f"misc/craft.pt", LANG)
            rec_model_path = download_or_load(f"misc/brainocr.pt", LANG)
            opt_fp = download_or_load(f"misc/ocr-opt.txt", LANG)

            reader = PororoReader(
                LANG,
                det_model_ckpt_fp=det_model_path,
                rec_model_ckpt_fp=rec_model_path,
                opt_fp=opt_fp,
                device=DEVICE,
            )
            reader.detector.to(DEVICE)
            reader.opt2val.update({
                'canvas_size': 2560,
                'mag_ratio': 1.0,
                'slope_ths': 0.1,
                'ycenter_ths': 0.5,
                'height_ths': 0.5,
                'width_ths': 0.5,
                'add_margin': 0.1,
                'min_size': 20,
                'text_threshold': 0.7,
                'low_text': 0.4,
                'link_threshold': 0.4,
            })
            _pororo_detector = reader
            logger.info("Pororo OCR detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Pororo OCR detector: {e}")
            _pororo_detector = None
    return _pororo_detector


def _convert_poly_to_box(points: np.ndarray) -> Box:
    rect = cv2.boundingRect(points.astype(np.int32))
    return Box(x=rect[0], y=rect[1], w=rect[2], h=rect[3])


def detect_paddle(img_bgr: np.ndarray) -> List[Box]:
    detector = get_paddle_detector()
    if detector is None:
        raise ImportError("PaddleOCR detector model is not loaded.")

    # det=True, rec=False로 실행 (det() 메소드 사용)
    result = detector.predict(img_bgr)
    if not result:
        return []

    boxes = []
    for poly_points in result:
        boxes.append(_convert_poly_to_box(np.array(poly_points)))
    return boxes


def detect_easyocr(img_bgr: np.ndarray) -> List[Box]:
    detector = get_easy_detector()
    if detector is None:
        raise ImportError("EasyOCR detector model is not loaded.")

    horizontal_list, free_list = detector.detect(img_bgr)

    boxes = []
    for (x_min, x_max, y_min, y_max) in horizontal_list:
        boxes.append(Box(x=int(x_min), y=int(y_min), w=int(x_max - x_min), h=int(y_max - y_min)))
    for poly_points in free_list:
        boxes.append(_convert_poly_to_box(np.array(poly_points)))

    return boxes


def detect_pororo(img_bgr: np.ndarray) -> List[Box]:
    """Pororo OCR을 사용하여 텍스트 영역을 탐지합니다."""
    detector = get_pororo_detector()
    if detector is None:
        raise ImportError("Pororo OCR detector model is not loaded.")

    # pororo.brainocr.Reader의 detect 메서드 사용
    horizontal_list, free_list = detector.detect(img_bgr, detector.opt2val)

    boxes = []
    # 1. Horizontal boxes
    for (x_min, x_max, y_min, y_max) in horizontal_list:
        boxes.append(Box(x=int(x_min), y=int(y_min), w=int(x_max - x_min), h=int(y_max - y_min)))

    # 2. Free-form boxes
    for poly_points in free_list:
        boxes.append(_convert_poly_to_box(np.array(poly_points)))

    return boxes


