import torch
import argparse
import os
import torch.optim as optim
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator
from modeling.losses import LeastSquareLossD
from modeling.losses import LeastSquareLossG
from modeling.losses import ContentLoss
from modeling.losses import ColorLoss
from modeling.losses import GramLoss
from modeling.vgg import get_vgg19
from dataloader import DataLoader
from tqdm import tqdm



def parse_args():
    desc = "get the mean values of  b,g,r on the whole dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao')
    parser.add_argument('--data-dir', type=str, default='/content')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--checkpoint-dir', type=str, default='/content/checkpoints')
    parser.add_argument('--save-interval', type=int, default=2)
    parser.add_argument('--lr-g', type=float, default=0.001)
    parser.add_argument('--lg-d', type=float, default=0.002)


    return parser.parse_args()


def main():
    args = parse_args()

    print("Init models...")

    G = Generator()
    D = Discriminator()
    vgg19 = get_vgg19()

    photo_loader = DataLoader(os.path.join(args.data_dir, 'train_photo'))
    anime_loader = DataLoader(os.path.join(args.data_dir, args.dataset, 'style'))
    anime_smooth_loader = DataLoader(os.path.join(args.data_dir, args.dataset, 'smooth'))
    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d)

    for e in range(args.epochs):
        print(f"Epoch {e}/{args.epochs}")

        for photo, _ in tqdm(photo_loader):
            anime, anime_gray = anime_loader.next()
            anime_smt, anime_gray_smt = anime_smooth_loader.next()

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake = G(photo)
            validity = D(fake)
            feature_fake = vgg19(fake)
            feature_anime = vgg19(anime_gray)
            feature_photo = vgg19(photo)

            loss_g = get_loss_G(fake, photo, validity, feature_fake, feature_anime, feature_photo)
            loss_g.backward()

            optimizer_g.step()

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()

            fake_d = D(fake.detach())
            real_d = D(photo)

            loss_d = get_loss_D(real_d, fake_d)
            loss_d.backward()

            optimizer_d.step()

        if e % args.save_interval == 0:
            save_weight(G, D)

        if e % args.plot_interval == 0:
            save_sample(G)








if __name__ == '__main__':
    main()