# -*- coding: utf-8 -*-
from typing import List, Tuple

import cv2
import torch
import sys
from logging import getLogger
from pathlib2 import Path
import numpy as np

from .detection import get_detector
from .utils import group_text_box, calculate_md5, download_and_unzip
from .config import *

#############################################
from .custom_detection import batch_get_textbox

#############################################


LOGGER = getLogger(__name__)


class CRAFT(object):

    def __init__(
        self, gpu=True, model_storage_directory="./weights", download_enabled=True
    ):

        self.download_enabled = download_enabled
        self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        self.user_network_directory = MODULE_PATH + "/user_network"
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu is False:
            self.device = "cpu"
            LOGGER.warning("Using CPU. Note: This module is much faster with a GPU.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            LOGGER.warning(
                "CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU."
            )
        elif gpu is True:
            self.device = "cuda"
        else:
            self.device = gpu

        # check and download detection model
        corrupt_msg = "MD5 hash mismatch, possible file corruption"
        detector_path = os.path.join(self.model_storage_directory, DETECTOR_FILENAME)

        if os.path.isfile(detector_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError(
                    "Missing %s and downloads disabled" % detector_path
                )
            LOGGER.warning(
                "Downloading detection model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(
                model_url["detector"][0],
                DETECTOR_FILENAME,
                self.model_storage_directory,
            )
            assert calculate_md5(detector_path) == model_url["detector"][1], corrupt_msg
            LOGGER.info("Download complete")
        elif calculate_md5(detector_path) != model_url["detector"][1]:
            if not self.download_enabled:
                raise FileNotFoundError(
                    "MD5 mismatch for %s and downloads disabled" % detector_path
                )
            LOGGER.warning(corrupt_msg)
            os.remove(detector_path)
            LOGGER.warning(
                "Re-downloading the detection model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(
                model_url["detector"][0],
                DETECTOR_FILENAME,
                self.model_storage_directory,
            )
            assert calculate_md5(detector_path) == model_url["detector"][1], corrupt_msg

        self.detector = get_detector(detector_path, self.device)

    def detect(
        self,
        img_list: List[np.ndarray],
        min_size=20,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1,
        text_detection_batch_size=1,
    ):

        batch_text_box = batch_get_textbox(
            self.detector,
            img_list,
            canvas_size,
            mag_ratio,
            text_threshold,
            link_threshold,
            low_text,
            False,
            self.device,
            text_detection_batch_size,
        )

        batch_horizontal_list = []
        batch_free_list = []
        for text_box in batch_text_box:
            horizontal_list, free_list = group_text_box(
                text_box, slope_ths, ycenter_ths, height_ths, width_ths, add_margin
            )
            batch_horizontal_list.append(horizontal_list)
            batch_free_list.append(free_list)

        return batch_horizontal_list, batch_free_list

    def convert_image_list2resized_image_list(self, image_list, size=512):
        resized_image_list = []
        for image in image_list:
            resized_image = cv2.resize(image, dsize=(size, size))
            resized_image_list.append(resized_image)
        return resized_image_list

    def back_to_original_point(
        self, image_list, batch_horizontal_list, batch_free_list, text_detection_resize
    ):
        for i in range(0, len(image_list)):
            image = image_list[i]
            new_height, new_width, new_channel = image.shape
            old_height, old_width, old_channel = (
                text_detection_resize,
                text_detection_resize,
                3,
            )
            height_ratio = new_height / old_height
            width_ratio = new_width / old_width

            horizontal_list = batch_horizontal_list[i]
            for j in range(0, len(horizontal_list)):
                [x_min, x_max, y_min, y_max] = horizontal_list[j]
                x_min, x_max = int(x_min * width_ratio), int(x_max * width_ratio)
                y_min, y_max = int(y_min * height_ratio), int(y_max * height_ratio)
                horizontal_list[j] = [x_min, x_max, y_min, y_max]

            free_list = batch_free_list[i]
            for j in range(0, len(free_list)):
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = free_list[j]
                x1, x2, x3, x4 = (
                    int(x1 * width_ratio),
                    int(x2 * width_ratio),
                    int(x3 * width_ratio),
                    int(x4 * width_ratio),
                )
                y1, y2, y3, y4 = (
                    int(y1 * height_ratio),
                    int(y2 * height_ratio),
                    int(y3 * height_ratio),
                    int(y4 * height_ratio),
                )
                free_list[j] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def detect_text_region(
        self,
        image_list: List[np.ndarray],
        text_detection_resize=512,
        text_detection_batch_size: int = 1,
        min_size=20,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1,
    ):
        batch_horizontal_list: List[List[Tuple[int, int, int, int]]]  # w_left, w_right, h_up, h_down

        resized_img_list = self.convert_image_list2resized_image_list(
            image_list, size=text_detection_resize
        )
        batch_horizontal_list, batch_free_list = self.detect(
            resized_img_list,
            min_size,
            text_threshold,
            low_text,
            link_threshold,
            canvas_size,
            mag_ratio,
            slope_ths,
            ycenter_ths,
            height_ths,
            width_ths,
            add_margin,
            text_detection_batch_size,
        )
        self.back_to_original_point(
            image_list, batch_horizontal_list, batch_free_list, text_detection_resize
        )

        return [batch_horizontal_list, batch_free_list]
