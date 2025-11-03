from django.conf import settings
import numpy as np
import cv2
import os
import io
import fitz
import time
import uuid
import shutil
import glob
from PIL import Image
import PIL
from logging import getLogger
from typing import Tuple

logger = getLogger(__name__)

PIL.Image.MAX_IMAGE_PIXELS = 933120000
from copy import copy
import base64
from random import random


SILENT_CLOSE = 2


def array2d_to_array1d(arr2d):
    """Converts 2D array to 1D array."""
    arr1d = []
    arr2d_shape = []
    for arr in arr2d:
        arr2d_shape.append(len(arr))
        for elem in arr:
            arr1d.append(elem)
    return arr1d, arr2d_shape


def array1d_to_array2d(arr1d, arr2d_shape):
    """Converts 1D array to 2D array."""
    if sum(arr2d_shape) != len(arr1d):
        raise Exception(
            "Sum of the shape array elements must be equal to the length of 1D array!"
        )
    arr2d = []
    for arr_len in arr2d_shape:
        arr = []
        for i in range(arr_len):
            temp = arr1d.pop(0)
            arr.append(temp)
        arr2d.append(arr)
    return arr2d


def check_if_transparent(img_path_list):
    transparent = [False for x in img_path_list]
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, -1)
        if len(img.shape) > 2:
            if img.shape[2] == 4:
                transparent[idx] = True
    return transparent


def replace_alpha_channel(img_path_list, inpainted_img_list, transparent_img_list):
    """
    Args:
        - img_path_list: list of image paths
        - inpainted_img_list: list of inpainted images
        - transparent_img_list: list of boolean values
    Returns:
        - list of images with alpha channel
    """
    new_inpainted_img_list: list[np.ndarray] = []
    for img_path, inpainted_img, is_transparent in zip(
        img_path_list, inpainted_img_list, transparent_img_list
    ):
        if is_transparent:
            original_img: np.ndarray = cv2.imread(img_path, -1)
            # 알파 채널 추출
            b, g, r, a = cv2.split(original_img)
            # 알파 채널 추가할 이미지 생성
            inpainted_img_transparent: np.ndarray = cv2.cvtColor(
                inpainted_img, cv2.COLOR_RGB2RGBA
            )
            # 이미지에 알파 채널 추가
            inpainted_img_transparent[:, :, 3] = a
            # 이미지 추가
            new_inpainted_img_list.append(inpainted_img_transparent)
        else:
            new_inpainted_img_list.append(inpainted_img)
    return new_inpainted_img_list


def cv2pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil2cv(im):
    """Converts PIL image to opencv image."""
    img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return img


def replace_images(
    bubbles_coordinates, erased_bubbles, extracted_bubble_masks, original_images
):
    """Replaces bubbles_coordinates of original image with erased_bubbles.
    Also returns bubble masks"""
    replaced_images = []
    bubble_masks = []
    for idx, original_image in enumerate(original_images):
        original_image_copy = copy(original_image)
        bubble_mask = np.zeros(original_image.shape)
        for bubble_idx, bubble in enumerate(erased_bubbles[idx]):
            up, down, left, right = bubbles_coordinates[idx][bubble_idx]
            original_image_copy[up:down, left:right] = bubble
            bubble_mask[up:down, left:right] = extracted_bubble_masks[idx][bubble_idx]
        replaced_images.append(original_image_copy)
        bubble_masks.append(bubble_mask.astype("uint8"))
    bubble_masks = make_transparent_masks(bubble_masks)
    return replaced_images, bubble_masks


def make_transparent_masks(bubble_masks):
    transparent_masks = []
    for mask in bubble_masks:
        mask_bw = mask[:, :, 0]
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
        mask_rgba[:, :, 3] = mask_bw
        transparent_masks.append(mask_rgba)
    return transparent_masks


def merge_mask_with_image(list_of_images, list_of_masks):
    output_list = []
    for img, mask in zip(list_of_images, list_of_masks):
        r_mask, g_mask, b_mask, a_mask = cv2.split(mask)
        r_mask[a_mask == 0] = 0
        r, g, b = cv2.split(img)
        r[r_mask == 255] = 255
        g[r_mask == 255] = 255
        b[r_mask == 255] = 255
        output = cv2.merge((r, g, b))
        output_list.append(output)
    return output_list


def resize_img(img, desired_height=1000, desired_width=1000):
    h, w = img.shape[:2]
    h_ratio = h / desired_height
    w_ratio = w / desired_width
    if (h_ratio < 1) and (w_ratio < 1):
        return img
    if h_ratio >= w_ratio:
        new_h = desired_height
        new_w = int(w / h_ratio)
    else:
        new_h = int(h / w_ratio)
        new_w = desired_width
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def resize_img_v2(
    img: np.ndarray, desired_height=1000, desired_width=1000
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    origin_h, origin_w = img.shape[:2]
    h_ratio = origin_h / desired_height
    w_ratio = origin_w / desired_width
    if (h_ratio < 1) and (w_ratio < 1):
        return {
            "origin_img": img,
            "resized": img,
            "origin_h": origin_h,
            "origin_w": origin_w,
            "new_h": origin_h,
            "new_w": origin_w,
        }
    if h_ratio >= w_ratio:
        new_h = desired_height
        new_w = int(origin_w / h_ratio)
    else:
        new_h = int(origin_h / w_ratio)
        new_w = desired_width
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # return resized, (origin_h, origin_w, new_h, new_w)
    return {
        "origin_img": img,
        "resized": resized,
        "origin_h": origin_h,
        "origin_w": origin_w,
        "new_h": new_h,
        "new_w": new_w,
    }


def show_images(list_of_images, duration=3):
    if isinstance(list_of_images[0], list):
        list_of_images, _ = array2d_to_array1d(list_of_images)
    for img in list_of_images:
        img = resize_img(img)
        cv2.imshow("Toontra_new Image", img)
        cv2.waitKey(duration * 1000)
    cv2.destroyAllWindows()


def save_images(list_of_images, image_path, output_dir="Toontra/demo_output"):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"
    for idx, img in enumerate(list_of_images):
        cv2.imwrite(output_dir + image_path[idx], img)


def save_single_image(img, path):
    cv2.imwrite(path, img)


def check_and_convert_image(img_path):
    # ADD PDF SUPPORT
    if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(img_path, -1)
        if img.ndim == 2:
            converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[-1] == 3:
                converted_img = img
            elif img.shape[-1] == 4:
                converted_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                converted_img = None
        else:
            converted_img = None
    else:
        converted_img = None
        print("Unsupported image format.\nToontra supported formarts: png, jpg.")
    return converted_img


def create_log(log_dir):
    DELETE_LOG_AFTER_N_DAYS = 30
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    created_on = time.time()

    for f in os.listdir(log_dir):
        f = os.path.join(log_dir, f)
        if os.stat(f).st_mtime < created_on - DELETE_LOG_AFTER_N_DAYS * 86400:
            os.remove(f)

    log = open(
        log_dir + time.strftime("%Y%m%d-%H%M%S", time.localtime(created_on)) + ".txt",
        "w",
    )
    log.write("-" * 68 + "\n")
    log.write("{:^19s}|{:^37s}|{:^10s}\n".format("TIME", "EVENT", "DURATION"))
    log.write("-" * 68 + "\n")
    log.write(
        "{:^19s}| {:36s}| {:>9}\n".format(
            time.strftime("%Y%m%d %H:%M:%S", time.localtime(created_on)),
            "Created Toontra_new Log",
            "",
        )
    )
    return log


def write_event(log, message, duration, error=False):
    time_stamp = time.time()
    if not error:
        time_str = "{:.3f}s".format(round(duration, 3))
    else:
        time_str = ""
    log.write(
        "{:^19s}| {:36s}| {:>8} \n".format(
            time.strftime("%Y%m%d %H:%M:%S", time.localtime(time_stamp)),
            message,
            time_str,
        )
    )


def get_file_list(dir):
    return glob.glob(dir + "/*")


def temp_save_enhanced_images(img_list):
    save_dir = str(uuid.uuid4())
    os.mkdir(save_dir)
    for idx, img in enumerate(img_list):
        cv2.imwrite(save_dir + "/" + str(idx + 1).zfill(4) + ".png", img)
    return save_dir


def create_random_temp_dir():
    random_temp_dir = str(uuid.uuid4()) + "/"
    os.mkdir(random_temp_dir)
    random_temp_dir = os.path.abspath(random_temp_dir) + "/"
    return random_temp_dir


def remove_temp_scans(save_dir=None):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def merge_images_into_one(img_dir=None, output_path="output.png"):
    if img_dir[-1] != "/":
        img_dir = img_dir + "/"
    images_path = glob.glob(img_dir + "*")
    if len(images_path) == 0:
        print(f"No images in {img_dir}.")
        return None
    images_ = [cv2.imread(i) for i in images_path]
    images_width = [i.shape[1] for i in images_]
    images_height = [i.shape[0] for i in images_]
    min_width = min(images_width)
    resize_coef = [min_width / i for i in images_width]
    new_heights = [int(i * j) for i, j in zip(images_height, resize_coef)]
    images = [
        cv2.resize(i, (min_width, h), cv2.INTER_AREA)
        for i, h in zip(images_, new_heights)
    ]
    output = np.vstack(images)
    cv2.imwrite(output_path, output)
    remove_temp_scans(img_dir)


def start_terminal(probability=0.05):
    if random() > probability:
        return
    title = b"IF9fX19fX18gIF9fX19fX18gIF9fX19fX18gIF9fICAgIF8gIF9fX19\
            fX18gIF9fX19fXyAgICBfX19fX19fICAgICAgICBfXyAgIF9fICBfX19f\
            X19fIAp8ICAgICAgIHx8ICAgICAgIHx8ICAgICAgIHx8ICB8ICB8IHx8I\
            CAgICAgIHx8ICAgIF8gfCAgfCAgIF8gICB8ICAgICAgfCAgfCB8ICB8fC\
            AgICAgICB8CnxfICAgICBffHwgICBfICAgfHwgICBfICAgfHwgICB8X3w\
            gfHxfICAgICBffHwgICB8IHx8ICB8ICB8X3wgIHwgICAgICB8ICB8X3wg\
            IHx8X19fXyAgIHwKICB8ICAgfCAgfCAgfCB8ICB8fCAgfCB8ICB8fCAgI\
            CAgICB8ICB8ICAgfCAgfCAgIHxffHxfIHwgICAgICAgfCAgICAgIHwgIC\
            AgICAgfCBfX19ffCAgfAogIHwgICB8ICB8ICB8X3wgIHx8ICB8X3wgIHx\
            8ICBfICAgIHwgIHwgICB8ICB8ICAgIF9fICB8fCAgICAgICB8ICAgICAg\
            fCAgICAgICB8fCBfX19fX198CiAgfCAgIHwgIHwgICAgICAgfHwgICAgI\
            CAgfHwgfCB8ICAgfCAgfCAgIHwgIHwgICB8ICB8IHx8ICAgXyAgIHwgIC\
            AgICAgfCAgICAgfCB8IHxfX19fXyAKICB8X19ffCAgfF9fX19fX198fF9\
            fX19fX198fF98ICB8X198ICB8X19ffCAgfF9fX3wgIHxffHxfX3wgfF9f\
            fCAgICAgICAgfF9fX3wgIHxfX19fX19ffAo="
    output = base64.b64decode(title).decode()
    print(output)


def get_merged_bbox_for_cluster(clusters_list):
    """
    Args:
        - clusters_list: list of list of list of coordinates
    Returns:
        - list of list of coordinates

    Process:
        1. Merge the coordinates
        2. Return the merged coordinates
    """
    logger.info(f"clusters_list: {clusters_list}")
    output = []
    for clusters in clusters_list:
        merged_bbox = []
        for cl in clusters:
            min_x1, max_x2, min_y1, max_y2 = cl[0]
            for c in cl:
                x1, x2, y1, y2 = c
                min_x1 = min(min_x1, x1)
                max_x2 = max(max_x2, x2)
                min_y1 = min(min_y1, y1)
                max_y2 = max(max_y2, y2)
            merged_bbox.append([min_x1, max_x2, min_y1, max_y2])
        output.append(merged_bbox)
    return output


def extract_scans(file=None):
    """THIS FUNCTION EXTRACTS SCANNED IMAGES FROM PDF.
    IT WILL NOT WORK PROPERLY WITH MULTIPLE IMAGES PER PAGE.
    ONE PAGE - ONE IMAGE."""
    save_dir = str(uuid.uuid4())

    save_dir = os.path.join(settings.MEDIA_ROOT, save_dir)
    os.mkdir(save_dir)
    pdf_file = fitz.open(file)
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images()
        img = page.get_images()[0]
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(save_dir + "/" + str(page_index + 1).zfill(4) + ".png")
    return save_dir
