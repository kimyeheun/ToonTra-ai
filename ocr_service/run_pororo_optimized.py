import os
import time

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from ocr_service.config.pororo_parameter import sfx_level1, sfx_level2, sfx_radial
from pororo import brainocr
from pororo.tasks.utils.download_utils import download_or_load


IMAGE_PATH = "../resource/강아지 어쩌구 웹툰/1.jpg"
SAVE_DIR   = "./result"
LANG       = "ko"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    """디렉토리 생성"""
    os.makedirs(path, exist_ok=True)


def cut_image_for_text_detection(img):
    """이미지를 적절한 크기로 분할"""
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
    """분할 이미지 좌표를 원본 이미지 좌표로 변환"""
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


def calculate_box_distance(box1, box2):
    """두 박스 간의 최소 거리 계산"""
    x1_1, x2_1, y1_1, y2_1 = box1
    x1_2, x2_2, y1_2, y2_2 = box2
    
    # 박스가 겹치는 경우
    if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
        return 0
    
    # X축 거리
    if x2_1 < x1_2:
        dx = x1_2 - x2_1
    elif x2_2 < x1_1:
        dx = x1_1 - x2_2
    else:
        dx = 0
    
    # Y축 거리
    if y2_1 < y1_2:
        dy = y1_2 - y2_1
    elif y2_2 < y1_1:
        dy = y1_1 - y2_2
    else:
        dy = 0
    
    return np.sqrt(dx**2 + dy**2)


def merge_nearby_boxes(boxes, distance_threshold=50, min_samples=1):
    """
    DBSCAN 클러스터링을 사용하여 인접한 박스들을 병합
    
    Args:
        boxes: 바운딩 박스 리스트 [[x1, x2, y1, y2], ...]
        distance_threshold: 박스를 같은 그룹으로 묶을 최대 거리
        min_samples: 클러스터를 형성하는 최소 박스 수
    
    Returns:
        merged_boxes: 병합된 바운딩 박스 리스트
    """
    if len(boxes) == 0:
        return []
    
    # 박스 중심점 계산
    centers = []
    for box in boxes:
        x1, x2, y1, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])
    
    centers = np.array(centers)
    
    # DBSCAN 클러스터링
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # 클러스터별로 박스 병합
    merged_boxes = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # 노이즈 포인트는 개별 처리
            indices = np.where(labels == label)[0]
            for idx in indices:
                merged_boxes.append(boxes[idx])
        else:
            # 같은 클러스터의 박스들을 하나로 병합
            indices = np.where(labels == label)[0]
            cluster_boxes = [boxes[i] for i in indices]
            
            # 병합: 최소/최대 좌표 찾기
            min_x1 = min(box[0] for box in cluster_boxes)
            max_x2 = max(box[1] for box in cluster_boxes)
            min_y1 = min(box[2] for box in cluster_boxes)
            max_y2 = max(box[3] for box in cluster_boxes)
            
            merged_boxes.append([min_x1, max_x2, min_y1, max_y2])
    
    return merged_boxes


def post_process_boxes_advanced(boxes, img_shape):
    """
    고급 후처리: 방사형 패턴 감지 및 병합
    
    Args:
        boxes: 원본 바운딩 박스 리스트
        img_shape: 이미지 shape (H, W, C)
    
    Returns:
        processed_boxes: 후처리된 바운딩 박스 리스트
    """
    if len(boxes) < 2:
        return boxes
    
    H, W = img_shape[:2]
    processed_boxes = []
    
    # 1. 박스 크기 기준으로 그룹화
    box_sizes = []
    for box in boxes:
        x1, x2, y1, y2 = box
        area = (x2 - x1) * (y2 - y1)
        box_sizes.append(area)
    
    # 비슷한 크기의 박스들을 그룹화
    size_threshold = np.median(box_sizes) * 0.5
    
    # 2. 방사형 패턴 감지
    center_x, center_y = W // 2, H // 2
    radial_boxes = []
    normal_boxes = []
    
    for box in boxes:
        x1, x2, y1, y2 = box
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        
        # 이미지 중심으로부터의 거리
        dist_from_center = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
        
        # 방사형 패턴 판단 (이미지 중심 근처에 있으면서 분산된 경우)
        if dist_from_center < min(H, W) * 0.7:
            radial_boxes.append(box)
        else:
            normal_boxes.append(box)
    
    # 3. 방사형 박스는 더 적극적으로 병합
    if radial_boxes:
        merged_radial = merge_nearby_boxes(
            radial_boxes, 
            distance_threshold=min(H, W) * 0.15,  # 이미지 크기 대비 15%
            min_samples=1
        )
        processed_boxes.extend(merged_radial)
    
    # 4. 일반 박스 병합
    if normal_boxes:
        merged_normal = merge_nearby_boxes(
            normal_boxes, 
            distance_threshold=min(H, W) * 0.08,  # 이미지 크기 대비 8%
            min_samples=1
        )
        processed_boxes.extend(merged_normal)
    
    return processed_boxes


def adaptive_parameter_selection(img):
    """
    이미지 특성에 따른 적응적 파라미터 선택
    
    Args:
        img: 입력 이미지 (BGR)
    
    Returns:
        selected_params: 선택된 파라미터 딕셔너리
    """
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 엣지 검출로 텍스트 복잡도 측정
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 이미지의 표준편차로 대비 측정
    std_dev = np.std(gray)
    
    print(f"이미지 분석 - Edge density: {edge_density:.3f}, Std dev: {std_dev:.2f}")
    
    # 파라미터 선택 로직
    if edge_density > 0.15 or std_dev > 50:
        # 복잡한 이미지 (많은 효과음)
        print("복잡한 이미지 감지 - sfx_level2 파라미터 사용")
        return sfx_level2
    elif edge_density > 0.08:
        # 중간 복잡도
        print("중간 복잡도 이미지 - sfx_radial 파라미터 사용")
        return sfx_radial
    else:
        # 단순한 이미지
        print("단순한 이미지 - sfx_level1 파라미터 사용")
        return sfx_level1


def draw_boxes_on_image(image_path: str, boxes, merged_boxes=None, save_dir: str = "./result") -> str:
    """
    원본 이미지에 바운딩 박스를 그려서 저장
    
    Args:
        image_path: 원본 이미지 경로
        boxes: 원본 바운딩 박스 리스트
        merged_boxes: 병합된 바운딩 박스 리스트 (옵션)
        save_dir: 저장 디렉토리
    
    Returns:
        저장된 이미지 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 원본 박스 (파란색)
    for bbox in boxes:
        x1, x2, y1, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # 병합된 박스 (노란색, 더 두껍게)
    if merged_boxes:
        for bbox in merged_boxes:
            x1, x2, y1, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 4)

    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    save_path = os.path.join(save_dir, f"{name}_detected_optimized{ext}")
    cv2.imwrite(save_path, img)
    print(f"결과 이미지 저장됨: {save_path}")
    return save_path


def load_pororo_reader(lang: str = LANG, device: str = DEVICE):
    """Pororo OCR Reader 로드"""
    print("필요한 모델 파일을 다운로드/확인 중...")
    det_model_path = download_or_load(f"misc/craft.pt", lang)
    rec_model_path = download_or_load(f"misc/brainocr.pt", lang)
    opt_fp         = download_or_load(f"misc/ocr-opt.txt", lang)

    reader = brainocr.Reader(
        lang,
        det_model_ckpt_fp=det_model_path,
        rec_model_ckpt_fp=rec_model_path,
        opt_fp=opt_fp,
        device=device,
    )
    reader.detector.to(device)
    print(f"Pororo Reader 로드 완료 (lang={lang}, device={device})")
    return reader


def run_pororo_detection_optimized(image_path: str, save_dir: str, use_adaptive=True, apply_post_process=True):
    """
    최적화된 OCR 텍스트 검출 실행
    
    Args:
        image_path: 입력 이미지 경로
        save_dir: 결과 저장 디렉토리
        use_adaptive: 적응적 파라미터 선택 사용 여부
        apply_post_process: 후처리 적용 여부
    
    Returns:
        all_boxes_global: 원본 이미지 기준 바운딩 박스 리스트
        processed_boxes: 후처리된 바운딩 박스 리스트
        save_path: 저장된 결과 이미지 경로
    """
    ensure_dir(save_dir)
    
    # 이미지 로드
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    H, W = img_bgr.shape[:2]
    
    print(f"\n{'='*60}")
    print(f"이미지 크기: {W}x{H}")
    
    # 파라미터 선택
    if use_adaptive:
        selected_params = adaptive_parameter_selection(img_bgr)
    else:
        selected_params = sfx_level1  # 기본값
    
    # 이미지 분할
    cuts, h, w, _ = cut_image_for_text_detection(img_bgr)
    print(f"총 {len(cuts)}개 컷으로 분할됨")
    
    # OCR 모델 로드 및 파라미터 적용
    ocr = load_pororo_reader()
    ocr.opt2val.update(selected_params)
    
    # 각 분할 이미지에서 텍스트 검출
    t0 = time.time()
    all_cut_boxes = []
    per_cut_counts = []

    for idx, cut_img in enumerate(cuts):
        horizontal_list, free_list = ocr.detect(cut_img, ocr.opt2val)
        all_cut_boxes.append(horizontal_list)
        per_cut_counts.append(len(horizontal_list))
        print(f"[cut {idx:02d}] 감지 박스={len(horizontal_list)}개")
    
    # 좌표 변환
    all_boxes_global = convert_cut_coordinates_to_full_image(all_cut_boxes, h, w)
    print(f"\n✅ 좌표 변환 완료: 총 {len(all_boxes_global)}개 박스")
    
    # 후처리 적용
    processed_boxes = all_boxes_global
    if apply_post_process and len(all_boxes_global) > 0:
        print("\n후처리 적용 중...")
        
        # 1차 병합: 기본 거리 기반
        processed_boxes = merge_nearby_boxes(
            all_boxes_global, 
            distance_threshold=min(H, W) * 0.1
        )
        print(f"1차 병합 후: {len(processed_boxes)}개 박스")
        
        # 2차 병합: 고급 방사형 패턴 처리
        processed_boxes = post_process_boxes_advanced(processed_boxes, img_bgr.shape)
        print(f"2차 병합 후: {len(processed_boxes)}개 박스")
    
    dt = time.time() - t0
    print(f"\n총 소요시간: {dt:.3f} sec")
    
    # 결과 이미지 저장
    save_path = draw_boxes_on_image(
        image_path, 
        all_boxes_global, 
        merged_boxes=processed_boxes if apply_post_process else None,
        save_dir=save_dir
    )
    
    # 통계 출력
    if processed_boxes:
        areas = [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2) in processed_boxes if x2 > x1 and y2 > y1]
        cover = (sum(areas) / float(H * W)) if H * W > 0 else 0.0
        print(f"\n최종 박스 수: {len(processed_boxes)}")
        print(f"커버리지: {cover*100:.2f}%")
    
    print(f"{'='*60}\n")
    
    return all_boxes_global, processed_boxes, save_path


# 다양한 설정으로 실험하는 함수
def experiment_with_parameters(image_path: str, save_dir: str = "./result_experiments"):
    """
    여러 파라미터 조합으로 실험 수행
    """
    from pororo_parameter_optimized import sfx_level1, sfx_level2, sfx_radial
    
    experiments = [
        ("sfx_level1", sfx_level1),
        ("sfx_level2", sfx_level2),
        ("sfx_radial", sfx_radial),
    ]
    
    results = []
    
    for name, params in experiments:
        print(f"\n{'='*60}")
        print(f"실험: {name}")
        print(f"{'='*60}")
        
        exp_dir = os.path.join(save_dir, name)
        ensure_dir(exp_dir)
        
        # OCR 실행
        ocr = load_pororo_reader()
        ocr.opt2val.update(params)
        
        img_bgr = cv2.imread(image_path)
        cuts, h, w, _ = cut_image_for_text_detection(img_bgr)
        
        all_cut_boxes = []
        for cut_img in cuts:
            horizontal_list, _ = ocr.detect(cut_img, ocr.opt2val)
            all_cut_boxes.append(horizontal_list)
        
        all_boxes = convert_cut_coordinates_to_full_image(all_cut_boxes, h, w)
        
        # 후처리
        processed = merge_nearby_boxes(all_boxes, distance_threshold=min(h, w) * 0.1)
        
        # 결과 저장
        save_path = draw_boxes_on_image(
            image_path, 
            all_boxes, 
            merged_boxes=processed,
            save_dir=exp_dir
        )
        
        results.append({
            'name': name,
            'original_boxes': len(all_boxes),
            'merged_boxes': len(processed),
            'reduction_rate': (1 - len(processed)/max(1, len(all_boxes))) * 100,
            'save_path': save_path
        })
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("실험 결과 요약")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['name']:15} | 원본: {r['original_boxes']:3}개 → 병합: {r['merged_boxes']:3}개 (감소율: {r['reduction_rate']:.1f}%)")
    
    return results


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {IMAGE_PATH}")
    
    # 기본 실행 (최적화 적용)
    print("=" * 60)
    print("최적화된 OCR 실행")
    print("=" * 60)
    run_pororo_detection_optimized(IMAGE_PATH, SAVE_DIR, use_adaptive=True, apply_post_process=True)
    
    # 실험 모드 (주석 해제하여 사용)
    # experiment_with_parameters(IMAGE_PATH, "./result_experiments")
