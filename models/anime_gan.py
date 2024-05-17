import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .conv_blocks import DownConv
from .conv_blocks import UpConv
from .conv_blocks import SeparableConv2D
from .conv_blocks import InvertedResBlock
from .conv_blocks import ConvBlock
from .layers import get_norm
from utils.common import initialize_weights


class GeneratorV1(nn.Module):
    def __init__(self, dataset=''):
        super(GeneratorV1, self).__init__()
        self.name = f'{self.__class__.__name__}_{dataset}'
        bias = False

        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(
        self,
        dataset=None,
        num_layers=1,
        use_sn=False,
        norm_type="instance",
    ):
        super(Discriminator, self).__init__()
        self.name = f'discriminator_{dataset}'
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        in_channels = channels
        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
                get_norm(norm_type, channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            in_channels = channels * 4
            channels *= 2

        channels *= 2
        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            get_norm(norm_type, channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, img):
        logits = self.discriminate(img)
        return logits
