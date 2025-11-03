# -*- coding: utf-8 -*-
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
import cv2
import numpy as np
import math


class ImageEnhancer:
    def __init__(self, opt):
        self.opt = opt
        # self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        loadnet = torch.load(
            self.opt.ImageEnhancer_weights, map_location=torch.device("cpu")
        )
        self.model.load_state_dict(loadnet["params_ema"], strict=True)
        if self.opt.cuda:
            self.model = self.model.to(self.opt.device)
        self.model.eval()
        self.scale = 4

    def enhance(self, img_path, scale=4, bw=False):
        ALPHA_CHANNEL_THRESHOLD = 10
        rgba_mode = False
        tile_size = self.opt.ImageEnhancer_tile_size
        tile_pad = self.opt.ImageEnhancer_tile_padding
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            rgba_mode = True
            mask = img[:, :, 3]
            mask[mask < ALPHA_CHANNEL_THRESHOLD] = 0
            mask[mask != 0] = 255
            mask = np.stack((mask,) * 3, axis=-1)  # 3 channel mask
            enhanced_mask = self.enhance_rgb_img(mask, scale=4, bw=True)

        img = img.astype(np.float32)
        img = img / 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.opt.device)
        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print("Error", error)
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        if bw:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = np.stack((output,) * 3, axis=-1)
        if rgba_mode:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA)
            output[:, :, 3] = enhanced_mask
        return output

    def enhance_v2(self, img_path, scale=4, bw=False):
        ALPHA_CHANNEL_THRESHOLD = 10
        rgba_mode = False
        tile_size = self.opt.ImageEnhancer_tile_size
        tile_pad = self.opt.ImageEnhancer_tile_padding
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:  # REPLACE RECURSION WITH INTERPOLATION
            rgba_mode = True
            mask = img[:, :, 3]
            mask[mask < ALPHA_CHANNEL_THRESHOLD] = 0
            mask[mask != 0] = 255
            mask = np.stack((mask,) * 3, axis=-1)  # 3 channel mask
            enhanced_mask = self.enhance_v2(mask, scale=4, bw=True)

        height, width, channels = img.shape
        output_image = (
            torch.zeros(1, channel, height * scale, width * scale)
            .float()
            .to(self.opt.device)
        )

        tile_data = SingleImageTilesDataset(
            img, tile_size=tile_size, tile_padding=tile_pad
        )
        tile_dataloader = torch.utils.data.DataLoader(
            tile_data, batch_size=16, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(tile_dataloader):
                batch = batch.to(self.opt.device)
                output_tile_batch = self.model(batch)

                output_image = construct_image(
                    output_img=output_image,
                    tiles=output_tile_batch,
                    batch_id=batch_idx,
                    original_img_shape=img.shape,
                    tiles_size=tile_size,
                    padding_size=tile_pad,
                    scale=scale,
                )
            output = output_tensor_to_image(output_image)
        return output

    def enhance_rgb_img(self, img, scale=4, bw=False):
        tile_size = self.opt.ImageEnhancer_tile_size
        tile_pad = self.opt.ImageEnhancer_tile_padding
        img = img.astype(np.float32)
        img = img / 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print("Error", error)
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        if bw:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        return output

    # ==============================================================================#
    #   EXPERIMENTAL VIDEO ENHANCEMENT
    # ==============================================================================#
    def enhance_video(self, video_path, output_video_path=None, scale=4, bw=False):
        """(EXPERIMENTAL VIDEO ENHANCEMENT) CURRENTLY ONLY SUPPORTS MP4 FORMAT"""

        tile_size = self.opt.ImageEnhancer_tile_size
        tile_pad = self.opt.ImageEnhancer_tile_padding

        if not output_video_path:
            output_video_path = video_path[:-4] + "_SR.mp4"

        video = cv2.VideoCapture(video_path)
        input_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_fps = video.get(cv2.CAP_PROP_FPS)
        input_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_magnitude = len(str(input_length))
        output_width = int(input_width * scale)
        output_height = int(input_height * scale)
        out_video = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            input_fps,
            (output_width, output_height),
        )

        current_frame = 0
        while True:
            ret, frame = video.read()
            if ret:
                current_frame += 1
                print(
                    "{}/{}".format(
                        str(current_frame).zfill(input_magnitude), input_length
                    ),
                    end="\r",
                )
                img = frame.astype(np.float32)
                img = img / 255
                img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
                img = img.unsqueeze(0).to(self.opt.device)
                batch, channel, height, width = img.shape
                output_height = height * scale
                output_width = width * scale
                output_shape = (batch, channel, output_height, output_width)
                output = img.new_zeros(output_shape)
                tiles_x = math.ceil(width / tile_size)
                tiles_y = math.ceil(height / tile_size)
                for y in range(tiles_y):
                    for x in range(tiles_x):
                        ofs_x = x * tile_size
                        ofs_y = y * tile_size
                        input_start_x = ofs_x
                        input_end_x = min(ofs_x + tile_size, width)
                        input_start_y = ofs_y
                        input_end_y = min(ofs_y + tile_size, height)
                        input_start_x_pad = max(input_start_x - tile_pad, 0)
                        input_end_x_pad = min(input_end_x + tile_pad, width)
                        input_start_y_pad = max(input_start_y - tile_pad, 0)
                        input_end_y_pad = min(input_end_y + tile_pad, height)
                        input_tile_width = input_end_x - input_start_x
                        input_tile_height = input_end_y - input_start_y
                        tile_idx = y * tiles_x + x + 1
                        input_tile = img[
                            :,
                            :,
                            input_start_y_pad:input_end_y_pad,
                            input_start_x_pad:input_end_x_pad,
                        ]
                        try:
                            with torch.no_grad():
                                output_tile = self.model(input_tile)
                        except RuntimeError as error:
                            print("Error", error)
                        output_start_x = input_start_x * scale
                        output_end_x = input_end_x * scale
                        output_start_y = input_start_y * scale
                        output_end_y = input_end_y * scale
                        output_start_x_tile = (
                            input_start_x - input_start_x_pad
                        ) * scale
                        output_end_x_tile = (
                            output_start_x_tile + input_tile_width * scale
                        )
                        output_start_y_tile = (
                            input_start_y - input_start_y_pad
                        ) * scale
                        output_end_y_tile = (
                            output_start_y_tile + input_tile_height * scale
                        )
                        output[
                            :,
                            :,
                            output_start_y:output_end_y,
                            output_start_x:output_end_x,
                        ] = output_tile[
                            :,
                            :,
                            output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile,
                        ]
                output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
                output = (output_img * 255.0).round().astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                if bw:
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    output = np.stack((output,) * 3, axis=-1)
                out_video.write(output)
            else:
                print("Completed video enhancement")
                break
        video.release()
        out_video.release()

        import os

        h264_output_path = output_video_path[:-4] + "_h264" + output_video_path[-4:]
        os.system(f"ffmpeg -i {output_video_path} -vcodec libx264 {h264_output_path}")
        os.remove(output_video_path)
        os.rename(h264_output_path, output_video_path)


# ==============================================================================#


class RRDBNet(nn.Module):
    def __init__(
        self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class SingleImageTilesDataset(torch.utils.data.Dataset):
    def __init__(self, img, tile_size, tile_padding):
        self.tile_size = tile_size
        self.tile_padding = tile_padding
        self.tiles = self.get_tiles(img)

    def get_tiles(self, img, scale=4):
        tiles = []

        img = img.astype(np.float32) / 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)

        batch, channel, height, width = img.shape
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                input_start_x = x * self.tile_size
                input_start_y = y * self.tile_size
                input_end_x = min(input_start_x + self.tile_size, width)
                input_end_y = min(input_start_y + self.tile_size, height)
                input_start_x_pad = max(input_start_x - self.tile_padding, 0)
                input_end_x_pad = min(input_end_x + self.tile_padding, width)
                input_start_y_pad = max(input_start_y - self.tile_padding, 0)
                input_end_y_pad = min(input_end_y + self.tile_padding, height)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                tiles.append(input_tile)
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]


def construct_image(
    output_img,
    tiles,
    batch_id,
    batch_size,
    original_img_shape,
    tiles_size=256,
    padding_size=10,
    scale=4,
):

    height, width, channel = original_img_shape
    output_height = height * scale
    output_width = width * scale

    for idx, tile in tiles:

        tiles_y_start_idx = (batch_id * batch_size + idx) // math.ceil(
            width / tiles_size
        )
        tiles_x_start_idx = (batch_id * batch_size + idx) % math.ceil(
            width / tiles_size
        )

        input_start_x = tiles_x_start_idx * tile_size
        input_end_x = min(input_start_x + tile_size, width)
        input_start_y = tiles_y_start_idx * tile_size
        input_end_y = min(input_start_y + tile_size, height)
        input_start_x_pad = max(input_start_x - padding_size, 0)
        input_end_x_pad = min(input_end_x + padding_size, width)
        input_start_y_pad = max(input_start_y - padding_size, 0)
        input_end_y_pad = min(input_end_y + padding_size, height)
        input_tile_width = input_end_x - input_start_x
        input_tile_height = input_end_y - input_start_y

        output_start_x = input_start_x * scale
        output_end_x = input_end_x * scale
        output_start_y = input_start_y * scale
        output_end_y = input_end_y * scale

        output_start_x_tile = (input_start_x - input_start_x_pad) * scale
        output_end_x_tile = output_start_x_tile + input_tile_width * scale
        output_start_y_tile = (input_start_y - input_start_y_pad) * scale
        output_end_y_tile = output_start_y_tile + input_tile_height * scale

        output_img[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = (
            tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]
        )

    return output_img


def output_tensor_to_image(output_tensor):
    output_img = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    output = (output_img * 255.0).round().astype(np.uint8)
    return output
