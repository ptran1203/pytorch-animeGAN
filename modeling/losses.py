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
                self.huber(image[2, ...], image_g[2, ...]))


class AnimeGanLoss:
    def __init__(self, args):
        self.adv_loss_g = LeastSquareLossG()
        self.adv_loss_d = LeastSquareLossD()
        self.content_loss = ContentLoss()
        self.gram_loss = GramLoss()
        self.color_loss = ColorLoss()
        self.adv_loss_g
        self.wadv = args.wadv
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol

    def compute_loss_G(self, gen_img, img, fake_d, gen_feat, anime_feat, img_feat):
        '''
        Compute loss for Generator

        @Arugments:
            - gen_img: generated image
            - img: image
            - fake_d: output of Discriminator given fake image
            - gen_feat: feature of fake image via VGG19
            - anime_feat: feature of anime image via VGG19
            - img_feat: feature of photo via VGG19

        @Returns:
            loss
        '''
        return (
            self.wadv * self.adv_loss_g(fake_d) +
            self.wcon * self.content_loss(img_feat, gen_feat) +
            self.wgra * self.gram_loss(anime_feat, gen_feat) +
            self.wcol * self.color_loss(img, gen_img)
        )

    def compute_loss_D(self, real_d, fake_d):
        return self.wadv * self.adv_loss_d(real_d, fake_d)
