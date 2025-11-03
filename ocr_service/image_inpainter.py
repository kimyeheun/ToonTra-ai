from PIL import Image
import numpy as np
import cv2
import os
import sys
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
from typing import List

from ocr_service.img_post_processor import PostProcessor

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",  # noqa
)


def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file


class SimpleLama:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.model = torch.jit.load(
            self.opt.ImageInpainter_weights, map_location=self.opt.device
        )
        self.model.eval()
        self.model.to(self.opt.device)
        self.device = self.opt.device

    def __call__(
        self, image: Image.Image or np.ndarray, mask: Image.Image or np.ndarray
    ):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            cur_res = Image.fromarray(cur_res)
            return cur_res


class ImageInpainter:
    def __init__(self, opt):
        self.opt = opt
        self.model = SimpleLama(opt)

    # 8비트 이미지로 변환
    def erase_text(
        self, img: np.ndarray, clusterred_coordinates: List[List[List[int]]]
    ) -> np.ndarray:
        """
        Args:
            - img: np.ndarray original image
            - clusterred_coordinates: list of list of coordinates
        Returns:
            - np.ndarray inpainted image

        Process:
            1. Create a mask from the clusterred coordinates
            2. Inpaint the image using the mask
            3. Return the inpainted image
        """

        original_h, original_w = img.shape[:2]

        ## imple 32비트 정수 이하로 이미지 리사이즈

        ## imple mask도 32비트 정수 이하로 리사이즈

        msk = np.zeros(img.shape[:2]).astype("uint8")
        # Text 영역을 사각형으로 만듦. 나중에 텍스트 영의 높이를 구해서 이미지 크기별 front size 공식을 통해서
        # 텍스트 원본 크기를 구할 수 있음.

        # # [기존 로직 - 사각형 마스크]
        # # clusterred_coordinates도 resized (원본 배열은 유지해야함)
        # for cl in clusterred_coordinates:
        #     for c in cl:
        #         x1, x2, y1, y2 = c
        #         cv2.rectangle(msk, (x1, y1), (x2, y2), color=(255), thickness=-1)

        # [새로운 로직 - 다각형 마스크 (convex hull)]
        # 클러스터당 하나의 볼록 다각형 생성 (말풍선 윤곽을 따라가는 정밀한 마스크)
        for cluster in clusterred_coordinates:
            # 클러스터 내 모든 textline의 꼭짓점 수집
            points = []
            for textline in cluster:
                x1, x2, y1, y2 = textline
                # 각 textline의 4개 꼭짓점 추가
                points.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            if len(points) > 0:
                # 모든 포인트를 포함하는 볼록 다각형(convex hull) 생성
                points_array = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points_array)
                cv2.fillPoly(msk, [hull], 255)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        mask = Image.fromarray(msk).convert("L")

        inpainted = self.model(image, mask)
        inpainted_cv = np.asarray(inpainted)
        inpainted_cv = cv2.cvtColor(inpainted_cv, cv2.COLOR_RGB2BGR)

        ## imple inpainted_cv를 원래 크기로 리사이즈
        return inpainted_cv[:original_h, :original_w]


class ImageInpainterWithPostProcessor(ImageInpainter):

    def __init__(self, opt, post_processor: PostProcessor):
        super().__init__(opt)
        self.post_processor = post_processor

    def erase_text(
        self, origin_img: np.ndarray, clusterred_coordinates: List[List[List[int]]]
    ) -> np.ndarray:
        erased_img = super().erase_text(origin_img, clusterred_coordinates)
        return erased_img
