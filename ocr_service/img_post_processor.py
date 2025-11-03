from logging import getLogger
from typing import List

import cv2
import numpy as np

from ocr_service.utils import util

logger = getLogger(__name__)


class PostProcessor:
    def __init__(self):
        pass

    def post_process(
        self,
        original_img: np.ndarray,
        erased_img: np.ndarray,
        clusterred_coordinates: List[List[List[int]]],
    ) -> np.ndarray:
        pass


class OriginalPreserver(PostProcessor):

    def __init__(self):
        super().__init__()

    def post_process(
        self,
        original_img: np.ndarray,
        erased_img: np.ndarray,
        clusterred_coordinates: List[List[List[int]]],
        blend_method: str = "gaussian",
        blend_size: int = 5,
    ) -> np.ndarray:
        """
        텍스트 영역은 지워진 이미지(erased_img)에서 가져오고,
        나머지 영역은 원본 이미지(original_img)에서 가져와서
        원본 이미지의 배경을 보존합니다.

        Args:
            blend_method: 블렌딩 방법 ("gaussian", "linear", "feather")
            blend_size: 블렌딩 영역 크기 (픽셀)
        """
        # 결과 이미지를 원본 이미지로 초기화 (배경 보존)
        result_img = original_img.copy()

        # 텍스트 영역의 바운딩 박스들을 병합
        bubble_boxes_group = util.get_merged_bbox_for_cluster([clusterred_coordinates])
        # 텍스트가 있는 영역만 지워진 이미지에서 가져오기
        logger.info(f"bubble_boxes_group: {bubble_boxes_group}")
        for bubble_boxes in bubble_boxes_group:
            for bubble_box in bubble_boxes:
                x1, x2, y1, y2 = bubble_box
                logger.info(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
                # 블렌딩 적용
                # if blend_method == "gaussian":
                #     result_img = self._blend_gaussian(
                #         result_img, erased_img, x1, x2, y1, y2, blend_size
                #     )
                # elif blend_method == "linear":
                #     result_img = self._blend_linear(
                #         result_img, erased_img, x1, x2, y1, y2, blend_size
                #     )
                # elif blend_method == "feather":
                #     result_img = self._blend_feather(
                #         result_img, erased_img, x1, x2, y1, y2, blend_size
                #     )
                # else:
                # 기본값: 단순 복사
                logger.info(f"Copying erased image to result image")
                result_img[y1:y2, x1:x2] = erased_img[y1:y2, x1:x2]

        return result_img

    def _blend_gaussian(self, result_img, erased_img, x1, x2, y1, y2, blend_size):
        """가우시안 블러를 이용한 자연스러운 블렌딩"""
        # 영역 추출
        original_region = result_img[y1:y2, x1:x2].copy()
        erased_region = erased_img[y1:y2, x1:x2].copy()

        # 마스크 생성 (중앙은 1, 가장자리는 0)
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        center_x, center_y = (x2 - x1) // 2, (y2 - y1) // 2

        # 거리 기반 마스크
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                max_dist = np.sqrt(center_y**2 + center_x**2)
                mask[i, j] = max(0, 1 - dist / (max_dist * 0.8))

        # 가우시안 블러 적용
        mask = cv2.GaussianBlur(
            mask, (blend_size * 2 + 1, blend_size * 2 + 1), blend_size
        )

        # 블렌딩 적용
        if len(original_region.shape) == 3:
            mask = np.stack([mask] * original_region.shape[2], axis=2)

        blended_region = original_region * (1 - mask) + erased_region * mask
        result_img[y1:y2, x1:x2] = blended_region.astype(np.uint8)

        return result_img

    def _blend_linear(self, result_img, erased_img, x1, x2, y1, y2, blend_size):
        """선형 그라데이션을 이용한 블렌딩"""
        # 영역 추출
        original_region = result_img[y1:y2, x1:x2].copy()
        erased_region = erased_img[y1:y2, x1:x2].copy()

        # 가장자리에서 중앙으로 갈수록 1에 가까워지는 마스크
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)

        # 가장자리에서 블렌딩
        for i in range(min(blend_size, (y2 - y1) // 2)):
            alpha = i / blend_size
            mask[i, :] = alpha
            mask[-(i + 1), :] = alpha
            mask[:, i] = alpha
            mask[:, -(i + 1)] = alpha

        # 블렌딩 적용
        if len(original_region.shape) == 3:
            mask = np.stack([mask] * original_region.shape[2], axis=2)

        blended_region = original_region * (1 - mask) + erased_region * mask
        result_img[y1:y2, x1:x2] = blended_region.astype(np.uint8)

        return result_img

    def _blend_feather(self, result_img, erased_img, x1, x2, y1, y2, blend_size):
        """Feather 효과를 이용한 블렌딩"""
        # 영역 추출
        original_region = result_img[y1:y2, x1:x2].copy()
        erased_region = erased_img[y1:y2, x1:x2].copy()

        # 마스크 생성 (중앙은 1, 가장자리는 0)
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)

        # 가장자리를 부드럽게 처리
        for i in range(min(blend_size, (y2 - y1) // 2)):
            alpha = i / blend_size
            # 상하좌우 가장자리
            mask[i, :] = alpha
            mask[-(i + 1), :] = alpha
            mask[:, i] = alpha
            mask[:, -(i + 1)] = alpha

        # 마스크를 부드럽게 만들기
        mask = cv2.GaussianBlur(mask, (blend_size + 1, blend_size + 1), blend_size / 3)

        # 블렌딩 적용
        if len(original_region.shape) == 3:
            mask = np.stack([mask] * original_region.shape[2], axis=2)

        blended_region = original_region * (1 - mask) + erased_region * mask
        result_img[y1:y2, x1:x2] = blended_region.astype(np.uint8)

        return result_img
