
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.conv_blocks import DownConv
from modeling.conv_blocks import UpConv
from modeling.conv_blocks import DsConv
from modeling.conv_blocks import InvertedResBlock
from modeling.conv_blocks import ConvBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            DownConv(128, 128),
            ConvBlock(128, 128),
            DsConv(128, 256),
            DownConv(256),
            ConvBlock(256, 256),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128),
            UpConv(128),
            DsConv(128, 128),
            ConvBlock(128, 64),
            UpConv(64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            *self.conv_blocks(32, level=1),
            *self.conv_blocks(128, level=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
        )


    @staticmethod
    def conv_blocks(in_channels, level):
        ins =  level * 64
        outs =  level * 128
        return [
            nn.Conv2d(in_channels, ins, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(ins, outs, kernel_size=3, stride=1),
            nn.InstanceNorm2d(outs),
            nn.LeakyReLU(),
        ]

    def forward(self, img):
        return self.discriminate(img)
