import os
import torch
import cv2
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from utils.image_processing import denormalize_input
from losses import LossSummary, AnimeGanLoss
from utils import load_checkpoint, save_checkpoint
from utils.common import set_lr


def gaussian_noise():
    gaussian_mean = torch.tensor(0.0)
    gaussian_std = torch.tensor(0.1)
    return torch.normal(gaussian_mean, gaussian_std)


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0),
    )


class Trainer:
    """
    Base Trainer class
    """

    def __init__(
        self,
        generator,
        discriminator,
        config
    ) -> None:
        self.G = generator
        self.D = discriminator
        self.cfg = config
        self.optimizer_g = optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.G.parameters(), lr=self.cfg.lr_d, betas=(0.5, 0.999))
        self.loss_tracker = LossSummary()
        self.device = torch.device(self.cfg.device)
        self.loss_fn = AnimeGanLoss(self.cfg, self.device)
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()
        self._init_working_dir()

    def _init_working_dir(self):
        """Init working directory for saving checkpoint, ..."""
        os.makedirs(self.cfg.exp_dir, exist_ok=True)
        self.checkpoint_path_G_init = os.path.join(self.cfg.exp_dir, "generator_init.pt")
        self.checkpoint_path_G = os.path.join(self.cfg.exp_dir, "generator.pt")
        self.checkpoint_path_D = os.path.join(self.cfg.exp_dir, "discriminator.pt")
        self.save_image_dir = os.path.join(self.cfg.exp_dir, "generated_images")

    def init_weight_G(self, weight: str):
        """Init Generator weight"""
        return load_checkpoint(self.G, weight)

    def init_weight_D(self, weight: str):
        """Init Discriminator weight"""
        return load_checkpoint(self.D, weight)

    def pretrain_generator(self, train_loader, start_epoch):
        init_losses = []
        set_lr(self.optimizer_g, self.cfg.init_lr)
        for epoch in range(start_epoch, self.cfg.init_epochs):
            # Train with content loss only
            
            pbar = tqdm(train_loader)
            for data in pbar:
                img = data["image"].to(self.device)

                self.optimizer_g.zero_grad()

                with torch.autocast(self.cfg.device, enabled=self.cfg.amp):
                    fake_img = self.G(img)
                    loss = self.loss_fn.content_loss_vgg(img, fake_img)

                self.scaler_g.scale(loss).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                pbar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')

            save_checkpoint(self.G, self.checkpoint_path_G_init, self.optimizer_g, epoch)
            self.generate_and_save(train_loader, subname='initg')
        set_lr(self.optimizer_g, self.cfg.lr_g)

    def train_epoch(self, epoch, train_loader):
        pbar = tqdm(train_loader, total=len(train_loader))
        for data in pbar:
            img = data["image"].to(self.device)
            anime = data["anime"].to(self.device)
            anime_gray = data["anime_gray"].to(self.device)
            anime_smt_gray = data["anime_smt_gray"].to(self.device)

            # ---------------- TRAIN D ---------------- #
            self.optimizer_d.zero_grad()

            with torch.autocast(self.cfg.device, enabled=self.cfg.amp):
                fake_img = self.G(img).detach()

            # Add some Gaussian noise to images before feeding to D
            if self.cfg.d_noise:
                fake_img += gaussian_noise()
                anime += gaussian_noise()
                anime_gray += gaussian_noise()
                anime_smt_gray += gaussian_noise()

            with torch.autocast(self.cfg.device, enabled=self.cfg.amp):
                fake_d = self.D(fake_img)
                real_anime_d = self.D(anime)
                real_anime_gray_d = self.D(anime_gray)
                real_anime_smt_gray_d = self.D(anime_smt_gray)

                loss_d = self.loss_fn.compute_loss_D(
                    fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)

            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()

            self.loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN G ---------------- #
            self.optimizer_g.zero_grad()

            with torch.autocast(self.cfg.device, enabled=self.cfg.amp):
                fake_img = self.G(img)
                fake_d = self.D(fake_img)

                adv_loss, con_loss, gra_loss, col_loss = self.loss_fn.compute_loss_G(
                    fake_img, img, fake_d, anime_gray)

            loss_g = adv_loss + con_loss + gra_loss + col_loss

            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            self.loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)
            pbar.set_description(self.loss_tracker.get_loss_description())

    def train(self, train_dataset: Dataset, start_epoch_g=0, start_epoch=0):
        """
        Train Generator and Discriminator.
        """
        # Set up dataloader
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
            # collate_fn=collate_fn,
        )
        self.G.to(self.device)
        self.D.to(self.device)

        self.pretrain_generator(data_loader, start_epoch_g)

        for epoch in range(start_epoch, self.cfg.epochs):
            self.train_epoch(epoch, data_loader)

            if epoch % self.cfg.save_interval == 0:
                save_checkpoint(self.G, self.checkpoint_path_G,self.optimizer_g, epoch)
                save_checkpoint(self.D, self.checkpoint_path_D, self.optimizer_d, epoch)
                self.generate_and_save(data_loader)

    def generate_and_save(
        self,
        loader,
        max_imgs=2,
        subname='gen'
    ):
        '''
        Generate and save images
        '''
        self.G.eval()

        max_iter = (max_imgs // self.cfg.batch_size) + 1
        fake_imgs = []

        for i, data in enumerate(loader):
            img = data["image"].to(self.device)
            with torch.no_grad():
                with torch.autocast(self.cfg.device, enabled=self.cfg.amp):
                    fake_img = self.G(img.cuda())
                fake_img = fake_img.detach().cpu().numpy()
                # Channel first -> channel last
                fake_img  = fake_img.transpose(0, 2, 3, 1)
                fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))

            if i + 1 == max_iter:
                break

        fake_imgs = np.concatenate(fake_imgs, axis=0)

        for i, img in enumerate(fake_imgs):
            save_path = os.path.join(self.save_image_dir, f'{subname}_{i}.jpg')
            cv2.imwrite(save_path, img[..., ::-1])
