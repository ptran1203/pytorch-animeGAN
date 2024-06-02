import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://pytorch.org/vision/0.12/_modules/torchvision/models/convnext.html
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm_type, channels):
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels)
    elif norm_type == "layer":
        # return LayerNorm2d
        return nn.GroupNorm(num_groups=1, num_channels=channels, affine=True)
        # return partial(nn.GroupNorm, 1, out_ch, 1e-5, True)
    else:
        raise ValueError(norm_type)
