import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.vgg import Vgg19
from util import gram, rgb_to_yuv_batch


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
        self.huber = nn.SmoothL1Loss()

    def forward(self, image, image_g):
        image = rgb_to_yuv_batch(image)
        image_g = rgb_to_yuv_batch(image_g)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, args):
        self.content_loss = ContentLoss().cuda()
        self.gram_loss = GramLoss().cuda()
        self.color_loss = ColorLoss().cuda()
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.vgg19 = Vgg19().cuda().eval()

    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):
        '''
        Compute loss for Generator

        @Arugments:
            - fake_img: generated image
            - img: image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image

        @Returns:
            loss
        '''
        with torch.no_grad():
            fake_feat = self.vgg19(fake_img)
            anime_feat = self.vgg19(anime_gray)
            img_feat = self.vgg19(img)

        return (
            self.wadvg * torch.mean(torch.square(fake_logit - 1.0)) +
            self.wcon * self.content_loss(img_feat, fake_feat) +
            self.wgra * self.gram_loss(anime_feat, fake_feat) +
            self.wcol * self.color_loss(img, fake_img)
        )

    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            1.7 * torch.mean(torch.square(real_anime_d - 1.0)) +
            1.7 * torch.mean(torch.square(fake_img_d)) +
            1.7 * torch.mean(torch.square(real_anime_gray_d)) +
            0.8 * torch.mean(torch.square(real_anime_smooth_gray_d))
        )


    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)