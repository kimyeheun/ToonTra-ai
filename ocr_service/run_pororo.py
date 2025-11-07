import os
import time
from typing import List

import cv2
import numpy as np
import torch

from ocr_service.config.pororo_parameter import basic
from pororo import brainocr
from pororo.tasks.utils.download_utils import download_or_load

IMAGE_PATH = "../resource/강아지 어쩌구 웹툰/1.jpg"
SAVE_DIR   = "./result"
LANG       = "ko"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def draw_boxes_on_image(image_path: str, boxes, save_dir: str = "./result") -> str:
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    for bbox in boxes:
        x1, x2, y1, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)

    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    save_path = os.path.join(save_dir, f"{name}_detected{ext}")
    cv2.imwrite(save_path, img)
    print(f"결과 이미지 저장됨: {save_path}")
    return save_path


def load_pororo_reader(lang: str = LANG, device: str = DEVICE):
    print("필요한 모델 파일을 다운로드/확인 중...(최초 실행 시 다소 소요)")
    det_model_path = download_or_load(f"misc/craft.pt", lang)
    rec_model_path = download_or_load(f"misc/brainocr.pt", lang)
    print(rec_model_path)
    opt_fp         = download_or_load(f"misc/ocr-opt.txt", lang)

    reader = brainocr.Reader(
        lang,
        det_model_ckpt_fp=det_model_path,
        rec_model_ckpt_fp=rec_model_path,
        opt_fp=opt_fp,
        device=device,
    )
    reader.detector.to(device)
    reader.opt2val.update(basic)
    print(f"Pororo Reader 로드 완료 (lang={lang}, device={device})")
    return reader

def run_pororo_detection(image_path: str, save_dir: str):
    ensure_dir(save_dir)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    H, W = img_bgr.shape[:2]

    # cuts 생성 (세로 긴 경우 w 높이로 자르고 w//2 오버랩)
    cuts, h, w, _ = cut_image_for_text_detection(img_bgr)
    print(f"총 {len(cuts)}개 컷으로 분할됨. (원본 H={H}, W={W})")

    ocr = load_pororo_reader()

    t0 = time.time()
    all_boxes_global: List[List[int]] = []
    per_cut_counts = []

    step = w // 2
    num_cuts = len(cuts)

    for idx, cut_img in enumerate(cuts):
        if w > h:
            y0 = 0
        else:
            y0 = (h - w) if (idx == num_cuts - 1) else (idx * step)

        horizontal_list, free_list = ocr.detect(cut_img, ocr.opt2val)

        boxes_this_cut = []
        for b in horizontal_list:
            # 원본 좌표로 y 오프셋 적용
            boxes_this_cut.append(b)

        all_boxes_global.extend(boxes_this_cut)
        per_cut_counts.append(len(boxes_this_cut))
        print(f"[cut {idx:02d}] y_offset={y0:5d}, 감지 박스={len(boxes_this_cut)}")

    dt = time.time() - t0
    print(f"\n총 박스 수: {len(all_boxes_global)}")
    print(f"컷별 평균 박스 수: {np.mean(per_cut_counts) if per_cut_counts else 0:.2f}")
    print(f"총 소요시간: {dt:.3f} sec  (컷당 {dt/max(1,len(cuts)):.3f} sec)")

    # ✅ 한 번만 호출 → 모든 컷 박스가 원본 이미지에 그려짐
    save_path = draw_boxes_on_image(image_path, all_boxes_global, save_dir=save_dir)

    # (옵션) 커버리지 리포트
    areas = [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2) in all_boxes_global if x2 > x1 and y2 > y1]
    cover = (sum(areas) / float(H * W)) if H * W > 0 else 0.0
    print(f"총 박스 면적 합: {sum(areas)}  |  이미지 면적 대비 커버리지: {cover*100:.2f}%")

    return all_boxes_global, save_path



if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {IMAGE_PATH}")
    run_pororo_detection(IMAGE_PATH, SAVE_DIR)