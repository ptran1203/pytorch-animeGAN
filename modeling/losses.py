import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.vgg import Vgg19
from util import gram, rgb_to_yuv_batch

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
        self.content_loss = nn.L1Loss().cuda()
        self.gram_loss = nn.L1Loss().cuda()
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
        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img)

        return [
            self.wadvg * torch.mean(torch.square(fake_logit - 1.0)),
            self.wcon * self.content_loss(img_feat.detach(), fake_feat),
            self.wgra * self.gram_loss(gram(anime_feat.detach()), gram(fake_feat)),
            self.wcol * self.color_loss(img.detach(), fake_img),
        ]

    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            torch.mean(torch.square(real_anime_d - 1.0)) +
            torch.mean(torch.square(fake_img_d)) +
            0.1 * torch.mean(torch.square(real_anime_gray_d)) +
            0.1 * torch.mean(torch.square(real_anime_smooth_gray_d))
        )


    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)


class LossSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_loss_G(self, adv, gram, color, content):
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_loss_D(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def avg_loss_D(self):
        return self._avg(self.loss_d_adv)


    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)