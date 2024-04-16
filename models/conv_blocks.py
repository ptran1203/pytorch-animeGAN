import torch.nn as nn
import torch.nn.functional as F
from utils.common import initialize_weights
from .layers import LayerNorm2d


class DownConv(nn.Module):

    def __init__(self, channels, bias=False):
        super(DownConv, self).__init__()

        self.conv1 = SeparableConv2D(channels, channels, stride=2, bias=bias)
        self.conv2 = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        out2 = self.conv2(out2)

        return out1 + out2


class UpConv(nn.Module):
    def __init__(self, channels, bias=False):
        super(UpConv, self).__init__()

        self.conv = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        out = self.conv(out)
        return out


class UpConvLNormLReLU(nn.Module):
    """Upsample Conv block with Layer Norm and Leaky ReLU"""
    def __init__(self, in_channels, out_channels, bias=False):
        super(UpConvLNormLReLU, self).__init__()

        self.conv_block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            bias=bias,
        )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        out = self.conv_block(out)
        return out

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, stride=1, bias=bias)
        # self.pad = 
        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.ins_norm1(out)
        out = self.activation1(out)

        out = self.pointwise(out)
        out = self.ins_norm2(out)

        return self.activation2(out)


class ConvBlock(nn.Module):
    """Stack of Conv2D + Norm + LeakyReLU"""
    def __init__(
        self,
        channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        norm_type="instance"
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if norm_type == "instance":
            self.ins_norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == "layer":
            self.ins_norm = LayerNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out



class InvertedResBlock(nn.Module):
    def __init__(
        self,
        channels=256,
        out_channels=256,
        expand_ratio=2,
        bias=False,
        norm_type="instance",
    ):
        super(InvertedResBlock, self).__init__()
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(
            channels,
            bottleneck_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.depthwise_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            groups=bottleneck_dim,
            stride=1,
            padding=1,
            bias=bias
        )
        self.conv = nn.Conv2d(
            bottleneck_dim,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=bias
        )

        if norm_type == "instance":
            self.ins_norm1 = nn.InstanceNorm2d(out_channels)
            self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        elif norm_type == "layer":
            # Keep var name as is for v1 compatibility.
            self.ins_norm1 = LayerNorm2d(bottleneck_dim)
            self.ins_norm2 = LayerNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        if out.shape[1] != x.shape[1]:
            # Only concate if same shape
            return out
        return out + x
