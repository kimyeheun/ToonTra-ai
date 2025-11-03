import torch

import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates

########
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List


########


# Dataset 상속
class ResizedImageDataset(Dataset):
    def __init__(self, resized_image_list: List[np.ndarray]):
        self.resized_image_list = resized_image_list

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.resized_image_list)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        img = self.resized_image_list[idx]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
        return img


def batch_get_textbox(
    detector,
    image_list,
    canvas_size,
    mag_ratio,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    text_detection_batch_size,
):
    batch_result = []
    batch_bboxes, batch_polys = batch_test_net(
        canvas_size,
        mag_ratio,
        detector,
        image_list,
        text_threshold,
        link_threshold,
        low_text,
        poly,
        device,
        text_detection_batch_size,
    )
    for cut_polys in batch_polys:
        result = []
        for box in cut_polys:
            poly = np.array(box).astype(np.int32).reshape((-1))
            result.append(poly)
        batch_result.append(result)

    return batch_result


def batch_test_net(
    canvas_size,
    mag_ratio,
    net,
    image_list,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    text_detection_batch_size,
    estimate_num_chars=False,
):
    resized_image_dataset = ResizedImageDataset(resized_image_list=image_list)
    resized_image_data_loader = DataLoader(
        resized_image_dataset, batch_size=text_detection_batch_size, shuffle=False
    )

    # 컷마다 box 와 poly 가 들어간다.
    batch_boxes = []
    batch_polys = []
    for batch_idx, batch_resized_images in enumerate(resized_image_data_loader):
        batch_resized_images = batch_resized_images.to(device)
        with torch.no_grad():
            y, feature = net(batch_resized_images)
        for i in range(0, len(y)):
            # make score and link map
            score_text = y[i, :, :, 0].cpu().data.numpy()
            score_link = y[i, :, :, 1].cpu().data.numpy()
            # Post-processing
            boxes, polys, _ = getDetBoxes(
                score_text,
                score_link,
                text_threshold,
                link_threshold,
                low_text,
                poly,
                estimate_num_chars,
            )

            boxes = adjustResultCoordinates(boxes, 1, 1)
            polys = adjustResultCoordinates(polys, 1, 1)

            for k in range(len(polys)):
                if polys[k] is None:
                    polys[k] = boxes[k]

            batch_boxes.append(boxes)
            batch_polys.append(polys)

    return batch_boxes, batch_polys
