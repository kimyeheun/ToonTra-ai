# -*- coding: utf-8 -*-

"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import easyocr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import math
from ocr_service.utils.korean_class import korean_classes #, korean_pororo_classes
from ocr_service.utils.util import array2d_to_array1d, array1d_to_array2d


class TextRecognizer:
    def __init__(self, opt, model_type):
        self.opt = opt
        self.converter = CTCLabelConverter(korean_classes)
        self.opt.num_class = len(self.converter.character)

        # model = Model(self.opt)
        # model.load_state_dict(torch.load(self.opt.saved_model, map_location=self.opt.device))
        # model = torch.nn.DataParallel(model).to(self.opt.device)
        # model.eval()

        self.model = self._set_model()
        self.model.eval()
        self.alignCollate = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
        )


    def _set_model(self):
        model = Model(self.opt)
        model = model.to(self.opt.device)

        checkpoint = torch.load(self.opt.saved_model, map_location=self.opt.device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        compatible_checkpoint = {}
        for k, v in checkpoint.items():
            if k in model.state_dict():
                compatible_checkpoint[k] = v
            else:
                compatible_checkpoint[k] = v

        model_state = model.state_dict()
        missing_keys = []
        unexpected_keys = []

        for k in model_state.keys():
            if k not in compatible_checkpoint:
                missing_keys.append(k)

        for k in compatible_checkpoint.keys():
            if k not in model_state:
                unexpected_keys.append(k)

        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")

        model.load_state_dict(compatible_checkpoint, strict=False)
        model = torch.nn.DataParallel(model).to(self.opt.device)
        return model

    def extract_text(self, list_of_pil_images):
        if isinstance(list_of_pil_images, Image.Image):
            list_of_pil_images = [[list_of_pil_images]]
        elif (list_of_pil_images
              and isinstance(list_of_pil_images, list)
              and isinstance(list_of_pil_images[0], Image.Image)):
            list_of_pil_images = [list_of_pil_images]

        list_of_pil_images, list_of_pil_images_shape = array2d_to_array1d(list_of_pil_images)
        output_strings_original = []
        for pil_images in list_of_pil_images:
            if isinstance(pil_images, Image.Image):
                pil_images = [pil_images]

            demo_data = RawDataset(pil_images=pil_images, opt=self.opt)
            demo_loader = torch.utils.data.DataLoader(
                demo_data,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.alignCollate,
                pin_memory=True,
            )
            output_strings = ""
            with torch.no_grad():
                for image_tensors, _ in demo_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(self.opt.device)
                    length_for_pred = torch.IntTensor(
                        [self.opt.batch_max_length] * batch_size
                    ).to(self.opt.device)
                    text_for_pred = (
                        torch.LongTensor(batch_size, self.opt.batch_max_length + 1)
                        .fill_(0)
                        .to(self.opt.device)
                    )
                    preds = self.model(image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, preds_size)
                    output_strings = " ".join(preds_str)
            output_strings_original.append(output_strings)
        output_strings = array1d_to_array2d(
            output_strings_original, list_of_pil_images_shape
        )
        return output_strings

    def extract_text_source_language(self, list_of_pil_images, source_lang="English"):
        """EXPERIMENTAL (Future Upgrade)
        Currently ONLY supports the following languages:
            - English
            - Portuguese
            - Vietnamese
            - Japanese
            - Ukrainian
            - Indonesian
        """

        LANGUAGES = {
            "English": "en",
            "Portuguese": "pt",
            "Vietnamese": "vi",
            "Japanese": "ja",
            "Ukrainian": "uk",
            "Indonesian": "id",
        }

        supported_languages = [
            "English",
            "Portuguese",
            "Vietnamese",
            "Japanese",
            "Ukrainian",
            "Indonesian",
        ]
        if source_lang.lower().capitalize() not in supported_languages:
            print(
                "Currently '{}' language is not supported.".format(source_lang.lower())
            )
            source_lang = LANGUAGES[source_lang.lower().capitalize()]
        reader = easyocr.Reader([source_lang], gpu=True)

        list_of_pil_images, list_of_pil_images_shape = array2d_to_array1d(
            list_of_pil_images
        )
        output_strings_original = []
        for pil_images in list_of_pil_images:
            output_string = []
            for pil_image in pil_images:
                img = np.asarray(pil_image)
                output = reader.readtext(img, paragraph="True")
                if len(output) != 0:
                    output_string.append(output[0][1])
            output_strings_original.append(" ".join(output_string))
        output_strings = array1d_to_array2d(
            output_strings_original, list_of_pil_images_shape
        )
        return output_strings


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            )

        return Pad_img


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RawDataset(Dataset):

    def __init__(self, pil_images, opt):
        self.opt = opt
        self.pil_images = pil_images
        self.nSamples = len(self.pil_images)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        import logging

        logger = logging.getLogger(__name__)

        logger.debug(
            f"pil_images 배열 크기: {len(self.pil_images)}, 현재 index: {index}"
        )

        try:
            if self.opt.rgb:
                img = self.pil_images[index].convert("RGB")  # for color image
            else:
                try:
                    img = self.pil_images[index].convert("L")
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)

                    logger.error(f"pil_images 배열 크기: {len(self.pil_images)}")
                    logger.error(f"현재 index: {index}")
                    logger.error(
                        f"pil_images[index] 타입: {type(self.pil_images[index])}"
                    )
                    logger.error(f"Error converting image to L: {e}")
                    logger.error(f"self.pil_images: {self.pil_images}")
                    logger.error(f"self.pil_images[index]: {self.pil_images[index]}")
                    logger.error(f"img: {self.pil_images[index]}")
                    # img = self.pil_images[index].convert("RGB")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new("RGB", (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new("L", (self.opt.imgW, self.opt.imgH))

        return img, ""


class CTCLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = [
            "[CTCblank]"
        ] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][: len(text)] = torch.LongTensor(text)
        return (
            batch_text.to(self.opt.device),
            torch.IntTensor(length).to(self.opt.device),
        )

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (
                    not (i > 0 and t[i - 1] == t[i])
                ):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)

            texts.append(text)
        return texts


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
                opt=self.opt,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            opt2val = {"rec_model_ckpt_fp": "baseline"}  # 기본값 설정

            self.FeatureExtraction = VGGFeatureExtractor(
            n_input_channels=opt.input_channel,
            n_output_channels=opt.output_channel,
            opt2val=opt2val
        )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                ),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, input, text, is_train=True):
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )

        return prediction


class TPS_SpatialTransformerNetwork(nn.Module):
    """Rectification Network of RARE, namely TPS based STN"""

    def __init__(self, F, I_size, I_r_size, I_channel_num=1, opt=None):
        """Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.opt = opt
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size, self.opt)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(
            batch_C_prime
        )  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2]
        )

        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(
                batch_I,
                build_P_prime_reshape,
                padding_mode="border",
                align_corners=True,
            )
        else:
            batch_I_r = F.grid_sample(
                batch_I, build_P_prime_reshape, padding_mode="border"
            )

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height)"""

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = (
            torch.from_numpy(initial_bias).float().view(-1)
        )

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(
            batch_size, self.F, 2
        )
        return batch_C_prime


class GridGenerator(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multipling T with P"""

    def __init__(self, F, I_r_size, opt):
        """Generate P_hat and inv_delta_C for later"""
        super(GridGenerator, self).__init__()
        self.opt = opt
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        ## for multi-gpu, you need register buffer
        self.register_buffer(
            "inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float()
        )  # F+3 x F+3
        self.register_buffer(
            "P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float()
        )  # n x F+3
        ## for fine-tuning with different image width, you may use below instead of self.register_buffer
        # self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        # self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

    def _build_C(self, F):
        """Return coordinates of fiducial points in I_r; C"""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """Return inv_delta_C which is needed to calculate T"""
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (
            np.arange(-I_r_width, I_r_width, 2) + 1.0
        ) / I_r_width  # self.I_r_width
        I_r_grid_y = (
            np.arange(-I_r_height, I_r_height, 2) + 1.0
        ) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(
            np.expand_dims(P, axis=1), (1, F, 1)
        )  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """Generate Grid from batch_C_prime [batch_size x F x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(self.opt.device)),
            dim=1,
        )  # batch_size x F+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C, batch_C_prime_with_zeros
        )  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2

class VGGFeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self,
                 n_input_channels: int = 1,
                 n_output_channels: int = 512,
                 opt2val=None):
        super(VGGFeatureExtractor, self).__init__()

        self.output_channel = [
            int(n_output_channels / 8),
            int(n_output_channels / 4),
            int(n_output_channels / 2),
            n_output_channels,
        ]  # [64, 128, 256, 512]

        rec_model_ckpt_fp = opt2val["rec_model_ckpt_fp"]
        if "baseline" in rec_model_ckpt_fp:
            self.ConvNet = nn.Sequential(
                nn.Conv2d(n_input_channels, self.output_channel[0], 3, 1, 1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64x16x50
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1,
                          1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 128x8x25
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1,
                          1),
                nn.ReLU(True),  # 256x8x25
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1,
                          1),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
                nn.Conv2d(self.output_channel[2],
                          self.output_channel[3],
                          3,
                          1,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.output_channel[3]),
                nn.ReLU(True),  # 512x4x25
                nn.Conv2d(self.output_channel[3],
                          self.output_channel[3],
                          3,
                          1,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.output_channel[3]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
                # nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24
                nn.ConvTranspose2d(self.output_channel[3],
                                   self.output_channel[3], 2, 2),
                nn.ReLU(True),
            )  # 512x4x50
        else:
            self.ConvNet = nn.Sequential(
                nn.Conv2d(n_input_channels, self.output_channel[0], 3, 1, 1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64x16x50
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1,
                          1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 128x8x25
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1,
                          1),
                nn.ReLU(True),  # 256x8x25
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1,
                          1),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
                nn.Conv2d(self.output_channel[2],
                          self.output_channel[3],
                          3,
                          1,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.output_channel[3]),
                nn.ReLU(True),  # 512x4x25
                nn.Conv2d(self.output_channel[3],
                          self.output_channel[3],
                          3,
                          1,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.output_channel[3]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
                # nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24
                nn.ConvTranspose2d(self.output_channel[3],
                                   self.output_channel[3], 2, 2),
                nn.ReLU(True),  # 512x4x50
                nn.ConvTranspose2d(self.output_channel[3],
                                   self.output_channel[3], 2, 2),
                nn.ReLU(True),
            )  # 512x4x50

    def forward(self, x):
        return self.ConvNet(x)


class ResNet_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.opt.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_size)
            .fill_(0)
            .to(self.opt.device)
        )
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size)
            .fill_(0)
            .to(self.opt.device),
            torch.FloatTensor(batch_size, self.hidden_size)
            .fill_(0)
            .to(self.opt.device),
        )

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(
                    text[:, i], onehot_dim=self.num_classes
                )
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[
                    0
                ]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = (
                torch.LongTensor(batch_size).fill_(0).to(self.opt.device)
            )  # [GO] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(self.opt.device)
            )

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(
            hidden_size, hidden_size
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(
            torch.tanh(batch_H_proj + prev_hidden_proj)
        )  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel
        concat_context = torch.cat(
            [context, char_onehots], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
