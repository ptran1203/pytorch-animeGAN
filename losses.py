import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vgg import Vgg19
from utils.image_processing import gram


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026]
        ]).float()

    def to(self, device):
        new_self = super(ColorLoss, self).to(device)
        new_self._rgb_to_yuv_kernel = new_self._rgb_to_yuv_kernel.to(device)
        return new_self

    def rgb_to_yuv(self, image):
        '''
        https://en.wikipedia.org/wiki/YUV

        output: Image of shape (H, W, C) (channel last)
        '''
        # -1 1 -> 0 1
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image,
            self._rgb_to_yuv_kernel,
            dims=([image.ndim - 3], [0]))

        return yuv_img

    def forward(self, image, image_g):
        image = self.rgb_to_yuv(image)
        image_g = self.rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, args, device):
        if isinstance(device, str):
            device = torch.device(device)

        self.content_loss = nn.L1Loss().to(device)
        self.gram_loss = nn.L1Loss().to(device)
        self.color_loss = ColorLoss().to(device)
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.vgg19 = Vgg19().to(device).eval()
        self.adv_type = args.gan_loss
        self.bce_loss = nn.BCELoss()

    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):
        '''
        Compute loss for Generator

        @Args:
            - fake_img: generated image
            - img: image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image

        @Returns:
            loss
        '''
        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img).detach()

        return [
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(anime_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img),
        ]

    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            self.adv_loss_d_real(real_anime_d) +
            self.adv_loss_d_fake(fake_img_d) +
            self.adv_loss_d_fake(real_anime_gray_d) +
            0.2 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )


    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)

    def adv_loss_d_real(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


    def adv_loss_g(self, pred):
        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


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

    def get_loss_description(self):
        avg_adv, avg_gram, avg_color, avg_content = self.avg_loss_G()
        avg_adv_d = self.avg_loss_D()
        return f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}'

    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)