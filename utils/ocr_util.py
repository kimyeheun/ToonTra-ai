from PIL import Image
import cv2
import numpy as np
from text_extract.data import Box

def clamp_box(box: Box, W: int, H: int) -> Box:
    x = max(0, min(box.x, W-1))
    y = max(0, min(box.y, H-1))
    w = max(1, min(box.w, W - x))
    h = max(1, min(box.h, H - y))
    return Box(x, y, w, h)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
