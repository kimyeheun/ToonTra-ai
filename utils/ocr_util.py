from typing import List

from PIL import Image
import cv2
import numpy as np
from text_extract.data import Box


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def ensure_clusters(boxes: List) -> List[List[List[int]]]:
    """
    입력이 이미 [[...],[...]] (클러스터들의 리스트)면 그대로,
    [x1,x2,y1,y2]들의 평면 리스트면 1개 클러스터로 감싸서 반환.
    """
    if not boxes:
        return []
    is_xyxy = lambda b: isinstance(b, (list, tuple)) and len(b) == 4
    # 평면 리스트인 경우
    if all(is_xyxy(b) for b in boxes):
        return [boxes]
    # 이미 클러스터 구조인 경우
    return boxes

def xyxy_cluster_to_xywh(cluster_xyxy: List[List[int]]) -> List[List[int]]:
    """한 클러스터 내의 xyxy 박스들을 xywh로 변환"""
    out = []
    for x1, x2, y1, y2 in cluster_xyxy:
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        if w > 0 and h > 0:
            out.append([x, y, w, h])
    return out
