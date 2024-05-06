# import torch
# from utils import compute_data_mean
# from models.anime_gan_v2 import GeneratorV2
# from models.anime_gan import GeneratorV1, Discriminator
# print(compute_data_mean('dataset/Hayao/style'))
import time
import cv2
import numpy as np
import torch
from enum import Enum

# https://github.com/jorge-pessoa/pytorch-colors/tree/master


# class ConvertCode(Enum):
RGB2LAB = 'RGB2LAB'
LAB2RGB = 'LAB2RGB'
RGB2YUV = 'RGB2YUV'
YUV2RGB = 'YUV2RGB'

_rgb2yuv_kernel = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.14714119, -0.28886916, 0.43601035],
    [0.61497538, -0.51496512, -0.10001026],
])

_yuv2rgb_kernel = torch.linalg.inv(_rgb2yuv_kernel)

_kernels = {
    RGB2YUV: ,
    YUV2RGB: torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026],
    ])
}


def get_kernel(kind: str):
    pass
    

def convert_color(image: torch.Tensor, code: str):
    """
    Args:
        image (torch.Tensor): Image tensor, can have shape: B x C x H x W
        code (str): convert kind, {'RGB2LAB', 'LAB2RGB'}

    References:
        + https://docs.opencv.org/4.3.0/de/d25/imgproc_color_conversions.html
    """

    return image


def color_transfer_torch(src, target):
    pass
    # # Convert to LAB space
    # src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    # target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # src_mean, src_std = get_mean_and_std(src)
    # target_mean, target_std = get_mean_and_std(target)

    # std_ratio = target_std / src_std
    # src_dtype = src.dtype

    # image = src.copy().astype(float)
    # image = (image - src_mean) * std_ratio + target_mean
    # image = image.astype(src_dtype)
    # image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)

    # return image


img1 = cv2.imread("/u02/phatth1/dataset/test/real/447.jpg")
img2 = cv2.imread("/u02/phatth1/dataset/Hayao/style/14.jpg")
img1 = torch.from_numpy(img1).unsqueeze(0).float()
img2 = torch.from_numpy(img2).unsqueeze(0).float()

yuv = img1 @ _kernels[RGB2YUV]

print(yuv)
