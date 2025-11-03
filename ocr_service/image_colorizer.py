# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import cv2
import numpy as np
import warnings
import opencv_transforms.functional as FF

warnings.simplefilter("ignore", UserWarning)

torch.backends.cudnn.benchmark = True


class ImageColorizer:
    def __init__(self, opt):
        self.opt = opt
        self.n_cluster = 9
        self.nc = 3 * (self.n_cluster + 1)
        self.netG = Sketch2Color(
            nc=self.nc, model_weights=self.opt.ImageColorizer_S2C_weights
        ).to(self.opt.device)
        self.netC2S = Color2Sketch(
            model_weights=self.opt.ImageColorizer_C2S_weights
        ).to(self.opt.device)

    def colorize(self, input_path, reference_path, output_path):
        """
        이미지 컬러화 처리:
        1. OpenCV로 이미지 로드 → RGB 변환
        2. 512x512 리사이즈 → 텐서 변환
        3. 컬러 팔레트 생성 (클러스터링 또는 기본 팔레트)
        4. AI 모델로 스케치 생성 + 컬러화
        5. 원본 크기로 복원 → 저장
        """
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        img_resize = cv2.resize(img, (512, 512), cv2.INTER_AREA)
        if reference_path is not None:
            color_palette = color_cluster(reference_path, nclusters=self.n_cluster)
        else:
            color_palette = []
            colors = [
                [227, 52, 47],
                [246, 153, 63],
                [255, 237, 74],
                [50, 50, 50],
                [77, 192, 181],
                [52, 144, 220],
                [101, 116, 205],
                [149, 97, 226],
                [250, 250, 250],
            ]
            for i in range(0, self.n_cluster):
                dominant_color = np.zeros(img_resize.shape, dtype="uint8")
                dominant_color[:, :, :] = colors[i]
                color_palette.append(dominant_color)

        # ==experimental==
        output_pallet = np.zeros([128, 128 * self.n_cluster, 3]).astype("uint8")
        for pal_idx, color_from_pallet in enumerate(color_palette):
            r, g, b = color_from_pallet[0, 0, :]
            output_pallet[:, 128 * pal_idx : 128 * (pal_idx + 1), 0] = b
            output_pallet[:, 128 * pal_idx : 128 * (pal_idx + 1), 1] = g
            output_pallet[:, 128 * pal_idx : 128 * (pal_idx + 1), 2] = r
        cv2.imwrite(output_path[:-4] + "_pallet" + output_path[-4:], output_pallet)
        # ================

        img_tensor = make_tensor(img_resize)
        for i in range(0, len(color_palette)):
            color = color_palette[i]
            color_palette[i] = make_tensor(color)
        with torch.no_grad():
            img_edge = (
                self.netC2S(img_tensor.unsqueeze(0).to(self.opt.device))
                .squeeze()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            output_edge = (((img_edge + 1) / 2) * 255).astype("uint8")
            cv2.imwrite(output_path[:-4] + "_sketch" + output_path[-4:], output_edge)
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)
            input_tensor = (
                torch.cat([img_edge] + color_palette, dim=0)
                .to(self.opt.device)
                .unsqueeze(dim=0)
            )
            output_tensor = self.netG(input_tensor)
            colorized = output_tensor.cpu()
            colorized = colorized.squeeze().permute(1, 2, 0).numpy()
            colorized = (((colorized + 1) / 2) * 255).astype("uint8")
            colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
            colorized = cv2.resize(colorized, (original_w, original_h), cv2.INTER_AREA)
        cv2.imwrite(output_path, colorized)

    def get_sketch(self, input_path, output_path):
        """
        스케치 생성:
        1. 이미지 512x512 리사이즈
        2. Color2Sketch 모델로 그레이스케일 스케치 생성
        3. 원본 크기로 복원 후 저장
        """
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        img_resize = cv2.resize(img, (512, 512), cv2.INTER_AREA)
        img_tensor = make_tensor(img_resize)
        with torch.no_grad():
            img_edge = (
                self.netC2S(img_tensor.unsqueeze(0).to(self.opt.device))
                .squeeze()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            output_edge = (((img_edge + 1) / 2) * 255).astype("uint8")
            output_edge = cv2.resize(
                output_edge, (original_w, original_h), cv2.INTER_AREA
            )
            output_edge = cv2.cvtColor(output_edge, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path, output_edge)


class Color2Sketch(nn.Module):
    def __init__(self, nc=3, pretrained=True, model_weights=None):
        super(Color2Sketch, self).__init__()
        self.model_weights = model_weights

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                # Build ResNet and change first conv layer to accept single-channel input
                self.layer1 = ResidualBlock(nc, 64, sample="down")
                self.layer2 = ResidualBlock(64, 128, sample="down")
                self.layer3 = ResidualBlock(128, 256, sample="down")
                self.layer4 = ResidualBlock(256, 512, sample="down")
                self.layer5 = ResidualBlock(512, 512, sample="down")
                self.layer6 = ResidualBlock(512, 512, sample="down")
                self.layer7 = ResidualBlock(512, 512, sample="down")

            def forward(self, input_image):
                # Pass input through ResNet-gray to extract features
                x0 = input_image  # nc * 256 * 256
                x1 = self.layer1(x0)  # 64 * 128 * 128
                x2 = self.layer2(x1)  # 128 * 64 * 64
                x3 = self.layer3(x2)  # 256 * 32 * 32
                x4 = self.layer4(x3)  # 512 * 16 * 16
                x5 = self.layer5(x4)  # 512 * 8 * 8
                x6 = self.layer6(x5)  # 512 * 4 * 4
                x7 = self.layer7(x6)  # 512 * 2 * 2

                return x1, x2, x3, x4, x5, x6, x7

        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                # Convolutional layers and upsampling
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample="up")

                self.Att6 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample="up")

                self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample="up")

                self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample="up")

                self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample="up")

                self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample="up")

                self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample="up")

                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input):  # , global_input):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input

                x = self.noise7(x7)
                x = self.layer7_up(x)  # 512 * 4 * 4

                x6 = self.Att6(g=x, x=x6)
                x = torch.cat((x, x6), dim=1)  # 1024 * 4 * 4
                x = self.layer6(x)  # 512 * 4 * 4
                x = self.noise6(x)
                x = self.layer6_up(x)  # 512 * 8 * 8

                x5 = self.Att5(g=x, x=x5)
                x = torch.cat((x, x5), dim=1)  # 1024 * 8 * 8
                x = self.layer5(x)  # 512 * 8 * 8
                x = self.noise5(x)
                x = self.layer5_up(x)  # 512 * 16 * 16

                x4 = self.Att4(g=x, x=x4)
                x = torch.cat((x, x4), dim=1)  # 1024 * 16 * 16
                x = self.layer4(x)  # 512 * 16 * 16
                x = self.noise4(x)
                x = self.layer4_up(x)  # 256 * 32 * 32

                x3 = self.Att3(g=x, x=x3)
                x = torch.cat((x, x3), dim=1)  # 512 * 32 * 32
                x = self.layer3(x)  # 256 * 32 * 32
                x = self.noise3(x)
                x = self.layer3_up(x)  # 128 * 64 * 64

                x2 = self.Att2(g=x, x=x2)
                x = torch.cat((x, x2), dim=1)  # 256 * 64 * 64
                x = self.layer2(x)  # 128 * 64 * 64
                x = self.noise2(x)
                x = self.layer2_up(x)  # 64 * 128 * 128

                x1 = self.Att1(g=x, x=x1)
                x = torch.cat((x, x1), dim=1)  # 128 * 128 * 128
                x = self.layer1(x)  # 64 * 128 * 128
                x = self.noise1(x)
                x = self.layer1_up(x)  # 32 * 256 * 256

                x = self.noise0(x)
                x = self.layer0(x)  # 3 * 256 * 256
                x = self.tanh(x)

                return x

        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained:
            checkpoint = torch.load(self.model_weights)
            self.load_state_dict(checkpoint["netG"], strict=True)

    def forward(self, inputs):
        encode = self.encoder(inputs)
        output = self.decoder(encode)

        return output


class Sketch2Color(nn.Module):
    def __init__(self, nc=3, pretrained=True, model_weights=None):
        super(Sketch2Color, self).__init__()
        self.model_weights = model_weights

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                # Build ResNet and change first conv layer to accept single-channel input
                self.layer1 = ResidualBlock(nc, 64, sample="down")
                self.layer2 = ResidualBlock(64, 128, sample="down")
                self.layer3 = ResidualBlock(128, 256, sample="down")
                self.layer4 = ResidualBlock(256, 512, sample="down")
                self.layer5 = ResidualBlock(512, 512, sample="down")
                self.layer6 = ResidualBlock(512, 512, sample="down")
                self.layer7 = ResidualBlock(512, 512, sample="down")

            def forward(self, input_image):
                # Pass input through ResNet-gray to extract features
                x0 = input_image  # nc * 256 * 256
                x1 = self.layer1(x0)  # 64 * 128 * 128
                x2 = self.layer2(x1)  # 128 * 64 * 64
                x3 = self.layer3(x2)  # 256 * 32 * 32
                x4 = self.layer4(x3)  # 512 * 16 * 16
                x5 = self.layer5(x4)  # 512 * 8 * 8
                x6 = self.layer6(x5)  # 512 * 4 * 4
                x7 = self.layer7(x6)  # 512 * 2 * 2

                return x1, x2, x3, x4, x5, x6, x7

        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                # Convolutional layers and upsampling
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample="up")

                self.Att6 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample="up")

                self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample="up")

                self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample="up")

                self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample="up")

                self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample="up")

                self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample="up")

                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input):  # , global_input):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input

                x = self.noise7(x7)
                x = self.layer7_up(x)  # 512 * 4 * 4

                x6 = self.Att6(g=x, x=x6)
                x = torch.cat((x, x6), dim=1)  # 1024 * 4 * 4
                x = self.layer6(x)  # 512 * 4 * 4
                x = self.noise6(x)
                x = self.layer6_up(x)  # 512 * 8 * 8

                x5 = self.Att5(g=x, x=x5)
                x = torch.cat((x, x5), dim=1)  # 1024 * 8 * 8
                x = self.layer5(x)  # 512 * 8 * 8
                x = self.noise5(x)
                x = self.layer5_up(x)  # 512 * 16 * 16

                x4 = self.Att4(g=x, x=x4)
                x = torch.cat((x, x4), dim=1)  # 1024 * 16 * 16
                x = self.layer4(x)  # 512 * 16 * 16
                x = self.noise4(x)
                x = self.layer4_up(x)  # 256 * 32 * 32

                x3 = self.Att3(g=x, x=x3)
                x = torch.cat((x, x3), dim=1)  # 512 * 32 * 32
                x = self.layer3(x)  # 256 * 32 * 32
                x = self.noise3(x)
                x = self.layer3_up(x)  # 128 * 64 * 64

                x2 = self.Att2(g=x, x=x2)
                x = torch.cat((x, x2), dim=1)  # 256 * 64 * 64
                x = self.layer2(x)  # 128 * 64 * 64
                x = self.noise2(x)
                x = self.layer2_up(x)  # 64 * 128 * 128

                x1 = self.Att1(g=x, x=x1)
                x = torch.cat((x, x1), dim=1)  # 128 * 128 * 128
                x = self.layer1(x)  # 64 * 128 * 128
                x = self.noise1(x)
                x = self.layer1_up(x)  # 32 * 256 * 256

                x = self.noise0(x)
                x = self.layer0(x)  # 3 * 256 * 256
                x = self.tanh(x)

                return x

        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained:
            checkpoint = torch.load(self.model_weights)
            self.load_state_dict(checkpoint["netG"], strict=True)

    def forward(self, inputs):
        encode = self.encoder(inputs)
        output = self.decoder(encode)

        return output


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class Conv2d_WS(nn.Conv2d):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_chan, out_chan, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return torch.nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sample=None):
        super(ResidualBlock, self).__init__()
        self.ic = in_channels
        self.oc = out_channels
        self.conv1 = Conv2d_WS(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = Conv2d_WS(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.convr = Conv2d_WS(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bnr = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sample = sample
        if self.sample == "down":
            self.sampling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.sample == "up":
            self.sampling = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.ic != self.oc:
            residual = self.convr(x)
            residual = self.bnr(residual)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.sample is not None:
            out = self.sampling(out)
        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            Conv2d_WS(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int),
        )

        self.W_x = nn.Sequential(
            Conv2d_WS(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int),
        )

        self.psi = nn.Sequential(
            Conv2d_WS(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def color_cluster(img_path, nclusters=9):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
    img_size = img.shape
    small_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    sample = small_img.reshape((-1, 3))
    sample = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    _, _, centers = cv2.kmeans(sample, nclusters, None, criteria, 10, flags)
    centers = np.uint8(centers)
    color_palette = []

    for i in range(0, nclusters):
        dominant_color = np.zeros(img_size, dtype="uint8")
        dominant_color[:, :, :] = centers[i]
        color_palette.append(dominant_color)

    return color_palette


def make_tensor(img):
    img = FF.to_tensor(img)
    img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img
