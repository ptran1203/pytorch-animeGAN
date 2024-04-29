
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.conv_blocks import DownConv
from models.conv_blocks import UpConv
from models.conv_blocks import SeparableConv2D
from models.conv_blocks import InvertedResBlock
from models.conv_blocks import ConvBlock
from utils.common import initialize_weights


class GeneratorV3(nn.Module):
    pass