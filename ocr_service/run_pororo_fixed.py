import os
import time
from typing import List

import cv2
import numpy as np
import torch

from ocr_service.config.pororo_parameter import basic, sfx_level1
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


def convert_cut_coordinates_to_full_image(cut_coordinates, img_height, img_width):
    # 분할이 없었던 경우 (가로 > 세로)
    if len(cut_coordinates) == 1:
        return cut_coordinates[0]
    
    _coordinates = []
    step = img_width // 2
    
    for idx, images in enumerate(cut_coordinates):
        if idx == len(cut_coordinates) - 1:
            y_offset = img_height - img_width
        else:
            y_offset = idx * step
        for line in images:
            x1, x2, y1, y2 = line
            new_cord = [x1, x2, y1 + y_offset, y2 + y_offset]
            _coordinates.append(new_cord)
    
    return _coordinates


def draw_boxes_on_image(image_path: str, boxes, save_dir: str = "./result") -> str:
    """
    원본 이미지에 바운딩 박스를 그려서 저장
    """
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
    """
    Pororo OCR Reader 로드
    
    Args:
        lang: 언어 코드 (기본값: "ko")
        device: 디바이스 ("cuda" 또는 "cpu")
    
    Returns:
        초기화된 Pororo Reader 객체
    """
    det_model_path = download_or_load(f"misc/craft.pt", lang)
    rec_model_path = download_or_load(f"misc/brainocr.pt", lang)
    opt_fp         = download_or_load(f"misc/ocr-opt.txt", lang)
    print(opt_fp)
    reader = brainocr.Reader(
        lang,
        det_model_ckpt_fp=det_model_path,
        rec_model_ckpt_fp=rec_model_path,
        opt_fp=opt_fp,
        device=device,
    )
    reader.detector.to(device)
    # TODO: 파라미터 수정
    reader.opt2val.update(sfx_level1)
    print(f"Pororo Reader 로드 완료 (lang={lang}, device={device})")
    return reader


def run_pororo_detection(image_path: str, save_dir: str):
    """
    이미지에서 텍스트 영역을 검출하고 결과를 저장
    
    Args:
        image_path: 입력 이미지 경로
        save_dir: 결과 저장 디렉토리
    
    Returns:
        all_boxes_global: 원본 이미지 기준 바운딩 박스 리스트
        save_path: 저장된 결과 이미지 경로
    
    처리 과정:
        1. 이미지 로드 및 분할 (세로로 긴 이미지인 경우)
        2. 각 분할 이미지에서 텍스트 검출
        3. 검출된 좌표를 원본 이미지 좌표로 변환
        4. 원본 이미지에 바운딩 박스 그리기
    """
    ensure_dir(save_dir)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    H, W = img_bgr.shape[:2]

    # ===== 1단계: 이미지 분할 =====
    cuts, h, w, _ = cut_image_for_text_detection(img_bgr)
    print(f"총 {len(cuts)}개 컷으로 분할됨. (원본 H={H}, W={W})")

    # ===== 2단계: OCR 모델 로드 =====
    ocr = load_pororo_reader()

    # ===== 3단계: 각 분할 이미지에서 텍스트 검출 =====
    t0 = time.time()
    
    # 각 cut별로 검출된 박스를 저장 (아직 좌표 변환 전)
    all_cut_boxes: List[List[List[int]]] = []
    per_cut_counts = []

    for idx, cut_img in enumerate(cuts):
        # cut 이미지에서 텍스트 영역 검출 (cut 이미지 기준 좌표)
        horizontal_list, free_list = ocr.detect(cut_img, ocr.opt2val)
        
        # 이 cut에서 검출된 박스들 저장
        all_cut_boxes.append(horizontal_list)
        per_cut_counts.append(len(horizontal_list))
        print(f"[cut {idx:02d}] 감지 박스={len(horizontal_list)}개")
    
    # ===== 4단계: 좌표 변환 =====
    # 모든 cut의 좌표를 원본 이미지 좌표로 변환
    all_boxes_global = convert_cut_coordinates_to_full_image(all_cut_boxes, h, w)
    print(f"\n✅ 좌표 변환 완료: 총 {len(all_boxes_global)}개 박스")

    dt = time.time() - t0
    print(f"컷별 평균 박스 수: {np.mean(per_cut_counts) if per_cut_counts else 0:.2f}")
    print(f"총 소요시간: {dt:.3f} sec  (컷당 {dt/max(1,len(cuts)):.3f} sec)")

    # ===== 5단계: 원본 이미지에 박스 그리기 =====
    save_path = draw_boxes_on_image(image_path, all_boxes_global, save_dir=save_dir)

    # ===== 6단계: 커버리지 리포트 (선택) =====
    areas = [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2) in all_boxes_global if x2 > x1 and y2 > y1]
    cover = (sum(areas) / float(H * W)) if H * W > 0 else 0.0
    print(f"총 박스 면적 합: {sum(areas)}  |  이미지 면적 대비 커버리지: {cover*100:.2f}%")

    return all_boxes_global, save_path


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {IMAGE_PATH}")
    run_pororo_detection(IMAGE_PATH, SAVE_DIR)
