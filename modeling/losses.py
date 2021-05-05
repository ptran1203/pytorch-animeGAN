import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn


_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
])

def gram(input):
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image)
    '''
    https://en.wikipedia.org/wiki/YUV
    '''

    return torch.mm(image, _rgb_to_yuv_kernel)


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


class RecontructionLoss(nn.Module):
    def __init__(self):
        super(RecontructionLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, feature, feature_g):
        return self.l1(gram(feature), gram(feature_g))