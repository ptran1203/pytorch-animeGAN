import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConv(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(DownConv, self).__init__()

        self.dsconv1 = DsConv(channels, channels, kernel_size=3, stride=2)
        self.dsconv2 = DsConv(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):

        out1 = self.dsconv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        out2 = self.dsconv2(out2)

        return out2 + out1


class UpConv(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(UpConv, self).__init__()

        self.dsconv = DsConv(channels, channels, kernel_size=3, stride=1)
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')

    def forward(self, x):
        out = self.upsample(x)
        out = self.dsconv(out)

        return out


class DsConv(nn.Module):

    def __init__(self, channels, out_channels, kernel_size=3, stride=1):
        super(DsConv, self).__init__()

        K = 1
        self.depthwise_conv = nn.Conv2d(channels, K * channels,
            kernel_size=kernel_size, groups=channels, stride=1, padding=1)

        self.ins_norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU()
        self.conv_block = ConvBlock(channels, out_channels, kernel_size=3, stride=stride)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)
        out = self.conv_block(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class InvertedResBlock(nn.Module):
    def __init__(self, channels=512, out_channels=256):
        super(InvertedResBlock, self).__init__()

        K = 1
        self.conv_block = ConvBlock(channels, 512, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(512, K * 512,
            kernel_size=3, groups=channels, stride=1, padding=1)
        self.conv = nn.Conv2d(512, out_channels,
            kernel_size=1, stride=1)

        self.ins_norm1 = nn.InstanceNorm2d(out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x
