
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm
from modeling.conv_blocks import DownConv
from modeling.conv_blocks import UpConv
from modeling.conv_blocks import SeparableConv2D
from modeling.conv_blocks import InvertedResBlock
from modeling.conv_blocks import ResnetBlock
from modeling.conv_blocks import ConvBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.name = 'generator'
        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            DownConv(128),
            ConvBlock(128, 128),
            SeparableConv2D(128, 256),
            DownConv(256),
            ConvBlock(256, 256),
        )

        Resblock = ResnetBlock
        # Resblock = InvertedResBlock

        self.res_blocks = nn.Sequential(
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
            Resblock(256, 256),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128),
            UpConv(128),
            SeparableConv2D(128, 128),
            ConvBlock(128, 128),
            UpConv(128),
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self,  args):
        super(Discriminator, self).__init__()
        self.name = 'discriminator'
        self.bias = True
        use_sn = args.use_sn
        image_size = 256
        batch_size = args.batch_size
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        feat_size = image_size // 4
        # print(f'{batch_size} * {feat_size} * {feat_size}',batch_size * feat_size * feat_size)
        self.linear = nn.Linear(feat_size * feat_size, 1)

    def forward(self, img):
        batch_size = img.shape[0]

        features = self.discriminate(img)

        return features
        
        logit = features.view(batch_size, - 1)

        return self.linear(logit)
