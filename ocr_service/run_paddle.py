import os
import cv2
from typing import List

from PIL import Image
import numpy as np


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def draw_boxes_on_image(image_path: str, boxes: List[List[int]], save_dir: str = "./result") -> str:
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 바운딩박스 그리기
    for bbox in boxes:
        cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 255, 255), 5)

    # 저장 경로 구성
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    save_path = os.path.join(save_dir, f"{name}_detected{ext}")

    cv2.imwrite(save_path, img)
    print(f"결과 이미지 저장됨: {save_path}")
    return save_path

def cut_image_for_text_detection(img):
    h, w = img.shape[:2]
    if w > h:
        return [img], h, w, img
    cuts = []
    for i in range(0, h - w // 2, w // 2):
        if (i + w) < h - 1:
            part = img[i : i + w, :, :]
        else:
            part = img[h - w :, :, :]
        cuts.append(part)
    return cuts, h, w, img


if __name__ == "__main__":
    image_path = "../resource/강아지 어쩌구 웹툰/1.jpg"
    image = Image.open(image_path)
    img_bgr = pil_to_bgr(image)
    cuts, h, w, img = cut_image_for_text_detection(img_bgr)

    # Paddle TextDetection 모델 초기화
    from paddleocr import TextDetection
    model = TextDetection(model_dir="/home/ubuntu/.paddlex/official_models/PP-OCRv5_server_det")

    # 감지 수행
    outputs = model.predict_iter(image_path)
    boxes = []

    for result in outputs:
        cluster = result.get('dt_polys', [])
        for box in cluster:
            for i in range(0, len(box), 2):
                start = box[i]
                end = box[i + 1]
                boxes.append([start[0], end[0], start[1], end[1]])
    print("paddle result")

    draw_boxes_on_image(image_path, boxes, save_dir="./result")
