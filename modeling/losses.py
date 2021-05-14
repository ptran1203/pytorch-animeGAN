import torch
import torch.nn.functional as F
import torch.nn as nn
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

        # print(image.shape, image_g.shape)

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, args):
        self.content_loss = ContentLoss()
        self.gram_loss = GramLoss()
        self.color_loss = ColorLoss()
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
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
            - anime_feat: feature of anime grayscale image via VGG19
            - img_feat: feature of photo via VGG19

        @Returns:
            loss
        '''
        return (
            self.wadvg * torch.mean(torch.square(fake_d - 1.0)) +
            self.wcon * self.content_loss(img_feat, gen_feat) +
            self.wgra * self.gram_loss(anime_feat, gen_feat) +
            self.wcol * self.color_loss(img, gen_img)
        )

    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            1.7 * torch.mean(torch.square(real_anime_d - 1.0)) +
            1.7 * torch.mean(torch.square(fake_img_d)) +
            1.7 * torch.mean(torch.square(real_anime_gray_d)) +
            0.8 * torch.mean(torch.square(real_anime_smooth_gray_d))
        )
