import torch
import torch.nn as nn



class DownConv(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(DownConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(channels, kernel_size*channels,
            kernel_size=kernel_size, group=channels, stride=1, padding=1)

        self.ins_norm1 = nn.InstanceNorm2d(channels)
        self.ins_norm2 = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU()
        self.conv_block = ConvBlock(channels, channels, kernel_size=1, stride=1)
        self.conv = nn.Conv2d(channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=1)

    def forward(self, x):

        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x




class UpConv(nn.Module):
    pass


class DsConv(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(DownConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(channels, kernel_size*channels,
            kernel_size=kernel_size, group=channels, stride=1, padding=1)

        self.ins_norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU()
        self.conv_block = ConvBlock(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)
        out = self.conv_block(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=1)
        self.ins_norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class InvertedResBlock(nn.Module):
    pass