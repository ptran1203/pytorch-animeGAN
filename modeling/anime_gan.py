
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
from modeling.conv_blocks import ConvBlock


def initialize_weights(net):
    total = 0
    init = 0
    return
    for m in net.modules():
        total += 1
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            init += 1
        except:
            pass

    print(f'Init {init}/{total}')


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
    def __init__(self,  use_sn=False):
        super(Discriminator, self).__init__()
        self.name = 'discriminator'

        if use_sn:
            self.discriminate = nn.Sequential(
                spectral_norm(nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False)),
                nn.LeakyReLU(0.2, True),
                *self.conv_blocks(32, level=1, use_sn=use_sn),
                *self.conv_blocks(128, level=2, use_sn=use_sn),
                spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False)),
            )
        else:
            self.discriminate = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False),
                nn.LeakyReLU(0.2, True),
                *self.conv_blocks(32, level=1),
                *self.conv_blocks(128, level=2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False),
            )


    @staticmethod
    def conv_blocks(in_channels, level, use_sn=False):
        ins =  level * 64
        outs =  level * 128

        conv1 = nn.Conv2d(in_channels, ins, kernel_size=3, stride=2, bias=False)
        conv2 = nn.Conv2d(ins, outs, kernel_size=3, stride=1, bias=False)

        if use_sn:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)

        return [
            conv1,
            nn.LeakyReLU(0.2, True),
            conv2,
            nn.InstanceNorm2d(outs),
            nn.LeakyReLU(0.2, True),
        ]

    def forward(self, img):
        x = self.discriminate(img)
        return x
