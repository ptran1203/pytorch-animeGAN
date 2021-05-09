import torch
import argparse
import os
import cv2
import numpy as np
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator
from modeling.losses import AnimeGanLoss
from modeling.losses import ContentLoss
from modeling.vgg import Vgg19
from dataset import AnimeDataSet
from util import show_images
from util import save_checkpoint
from util import load_checkpoint
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Hayao')
    parser.add_argument('--data-dir', type=str, default='/content')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint-dir', type=str, default='/content/checkpoints')
    parser.add_argument('--save-image-dir', type=str, default='/content/images')
    parser.add_argument('--continu', action='store_true')
    parser.add_argument('--display-image', type=bool, default=True)
    parser.add_argument('--save-interval', type=int, default=2)
    parser.add_argument('--debug-samples', type=int, default=0)
    parser.add_argument('--lr-g', type=float, default=0.001)
    parser.add_argument('--lr-d', type=float, default=0.002)
    parser.add_argument('--wadv', type=float, default=300.0, help='Adversarial loss weight')
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=3, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=10, help='Color loss weight')

    return parser.parse_args()


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0),
    )


def check_params(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found {data_path}')

    if not os.path.exists(args.save_image_dir):
        print(f'* {args.save_image_dir} does not exist, creating...')
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'* {args.checkpoint_dir} does not exist, creating...')
        os.makedirs(args.checkpoint_dir)


def save_samples(generator, loader, args, max_imgs=2):
    '''
    Generate and save images after a number of epochs
    '''
    def toint(img):
        img = img * 127.5 + 127.5
        img = img.astype(np.int16)
        return img

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []
    real_imgs = []

    for i, (img, *_) in enumerate(loader):
        fake_img = generator(img.cuda())
        fake_img = fake_img.detach().cpu().numpy()
        # Channel first -> channel last
        fake_img  = fake_img.transpose(0, 2, 3, 1)
        fake_imgs.append(toint(fake_img))
        real_imgs.append(
            toint(img.permute(0, 2, 3 ,1).detach().cpu().numpy()))

        if i + 1== max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)
    real_imgs = np.concatenate(real_imgs, axis=0)

    if args.display_image:
        show_images(np.concatenate([real_imgs, fake_imgs]), save=False)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f'gen_{i}.jpg')
        cv2.imwrite(save_path, img)


def main():
    args = parse_args()

    check_params(args)

    print("Init models...")

    G = Generator().cuda()
    D = Discriminator().cuda()
    vgg19 = Vgg19().cuda()
    
    loss_fn = AnimeGanLoss(args)

    # Create DataLoader
    num_workers = cpu_count()
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d)

    start_e = 0
    if args.continu:
        try:
            load_checkpoint(G, args)
            start_e = load_checkpoint(D, args)
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)


    for e in range(start_e, args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)

        if e < args.init_epochs:
            # Train with content loss only
            for img, *_ in bar:
                img = img.cuda()
                
                optimizer_g.zero_grad()

                fake_img = G(img)
                fake_feat = vgg19(fake_img)
                img_feat = vgg19(img)

                loss_g = ContentLoss()(img_feat, fake_feat)
                loss_g.backward()
                optimizer_g.step()

                bar.set_description(f'[Init Training G] content loss: {loss_g:2f}')

            continue

        for img, anime, anime_gray, anime_smt_gray in bar:

            # To cuda
            img = img.cuda()
            anime = anime.cuda()
            anime_gray = anime_gray.cuda()
            anime_smt_gray = anime_smt_gray.cuda()

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)
            fake_feat = vgg19(fake_img)
            anime_feat = vgg19(anime_smt_gray)
            img_feat = vgg19(img)

            loss_g = loss_fn.compute_loss_G(
                fake_img, img, fake_d, fake_feat, anime_feat, img_feat)

            loss_g.backward()

            optimizer_g.step()

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()

            fake_d = D(fake_img.detach())
            real_anime_d = D(anime)
            real_anime_gray_d = D(anime_gray)
            real_anime_smt_gray_d = D(anime_smt_gray)

            loss_d = loss_fn.compute_loss_D(
                fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)

            loss_d.backward()

            optimizer_d.step()

            # Set bar desc
            loss_g = loss_g.detach().cpu().numpy()
            loss_d = loss_d.detach().cpu().numpy()
            bar.set_description(f'loss G: {loss_g:2f}, loss D: {loss_d:2f}')

        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_samples(G, data_loader, args)


if __name__ == '__main__':
    main()