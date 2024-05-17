import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vgg import Vgg19
from utils.image_processing import gram


def to_gray_scale(image):
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_color.py#L33
    # Image are assum in range 1, -1
    image = (image + 1.0) / 2.0 # To [0, 1]
    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.unsqueeze(dim=-3)
    l_img = l_img.to(image.dtype)
    l_img = l_img.expand(image.shape)
    l_img = l_img / 0.5 - 1.0 # To [-1, 1]
    return l_img


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        # self._rgb_to_yuv_kernel = torch.tensor([
        #     [0.299, -0.14714119, 0.61497538],
        #     [0.587, -0.28886916, -0.51496512],
        #     [0.114, 0.43601035, -0.10001026]
        # ]).float()

        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
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
        image = image.permute(0, 2, 3, 1) # To channel last

        yuv_img = image @ self._rgb_to_yuv_kernel.T

        return yuv_img

    def forward(self, image, image_g):
        image = self.rgb_to_yuv(image)
        image_g = self.rgb_to_yuv(image_g)
        # After convert to yuv, both images have channel last
        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )


class AnimeGanLoss:
    def __init__(self, args, device, gray_adv=False):
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
        self.wtvar = args.wtvar
        # If true, use gray scale image to calculate adversarial loss
        self.gray_adv = gray_adv
        self.vgg19 = Vgg19().to(device).eval()
        self.adv_type = args.gan_loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):
        '''
        Compute loss for Generator

        @Args:
            - fake_img: generated image
            - img: real image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image

        @Returns:
            - Adversarial Loss of fake logits
            - Content loss between real and fake features (vgg19)
            - Gram loss between anime and fake features (Vgg19)
            - Color loss between image and fake image
            - Total variation loss of fake image
        '''
        fake_feat = self.vgg19(fake_img)
        gray_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img)
        # fake_gray_feat = self.vgg19(to_gray_scale(fake_img))

        return [
            # Want to be real image.
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(gray_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img),
            self.wtvar * self.total_variation_loss(fake_img)
        ]

    def compute_loss_D(
        self,
        fake_img_d,
        real_anime_d,
        real_anime_gray_d,
        real_anime_smooth_gray_d=None
    ):
        if self.gray_adv:
            # Treat gray scale image as real
            return (
                self.adv_loss_d_real(real_anime_gray_d)
                + self.adv_loss_d_fake(fake_img_d)
                + 0.3 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
            )
        else:
            return (
                # Classify real anime as real
                self.adv_loss_d_real(real_anime_d)
                # Classify generated as fake
                + self.adv_loss_d_fake(fake_img_d)
                # Classify real anime gray as fake
                # + self.adv_loss_d_fake(real_anime_gray_d)
                # Classify real anime as fake
                # + 0.1 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
            )

    def total_variation_loss(self, fake_img):
        """
        A smooth loss in fact. Like the smooth prior in MRF.
        V(y) = || y_{n+1} - y_n ||_2
        """
        # Channel first -> channel last
        fake_img = fake_img.permute(0, 2, 3, 1)
        def _l2(x):
            # sum(t ** 2) / 2
            return torch.sum(x ** 2) / 2

        dh = fake_img[:, :-1, ...] - fake_img[:, 1:, ...]
        dw = fake_img[:, :, :-1, ...] - fake_img[:, :, 1:, ...]
        return _l2(dh) / dh.numel() + _l2(dw) / dw.numel()

    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)
        feature_loss = self.content_loss(feat, re_feat)
        content_loss = self.content_loss(image, recontruction)
        return feature_loss# + 0.5 * content_loss

    def adv_loss_d_real(self, pred):
        """Push pred to class 1 (real)"""
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, pred):
        """Push pred to class 0 (fake)"""
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_g(self, pred):
        """Push pred to class 1 (real)"""
        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.ones_like(pred))

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
