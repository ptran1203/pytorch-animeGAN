import torch
import torch.nn.functional as F
import torch.nn as nn
from util import gram, rgb_to_yuv


class LeastSquareLossD(nn.Module):
    # https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan

    def __init__(self, a=0, b=1):
        super(LeastSquareLossD, self).__init__()
        self.a = a
        self.b = b

    def forward(self, pred_d, pred_g):
        return 0.5 * torch.mean((pred_d - self.b) ** 2) + 0.5 * torch.mean((pred_g - self.a) ** 2)


class LeastSquareLossG(nn.Module):
    def __init__(self, c=1):
        super(LeastSquareLossG, self).__init__()
        self.c = c

    def forward(self, pred_g):
        return 0.5 * torch.mean((pred_g - c) ** 2)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, feature, feature_g):
        return self.l1(feature, feature_g)


class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, feature, feature_g):
        return self.l1(gram(feature), gram(feature_g))


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.HuberLoss()

    def forward(self, image, image_g):
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        return (self.l1(image[0, ...], image_g[0, ...]) +
                self.huber(image[1, ...], image_g[1, ...]) +
                self.huber(image[2, ...], image_g[2, ...])

