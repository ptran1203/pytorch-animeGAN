import torch
import argparse
import os
import cv2
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator
from modeling.losses import AnimeGanLoss
from modeling.vgg import get_vgg19
from dataset import AnimeDataSet
from util import show_images
from util import save_checkpoint
from util import load_checkpoint
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Hayao')
    parser.add_argument('--data-dir', type=str, default='/content')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint-dir', type=str, default='/content/checkpoints')
    parser.add_argument('--save-image-dir', type=str, default='/content/images')
    parser.add_argument('--display-image', type=bool, default=True)
    parser.add_argument('--save-interval', type=int, default=2)
    parser.add_argument('--lr-g', type=float, default=0.001)
    parser.add_argument('--lr-d', type=float, default=0.002)
    parser.add_argument('--wadv', type=float, default=300.0, help='Adversarial loss weight')
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=3, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=10, help='Color loss weight')

    return parser.parse_args()


def collate_fn(batch):
    img, img_gray = zip(*batch)
    return torch.stack(img, 0), torch.stack(img_gray, 0)


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


def save_samples(generator, loader, args, max_imgs=3):
    '''
    Generate and save images after a number of epochs
    '''
    max_iter = max_imgs // args.batch_size
    fake_imgs = []

    for i, (img, _) in enumerate(loader):
        fake_img = generator(img)
        fake_img = fake_img.detach().cpu().numpy()
        # Channel first -> channel last
        fake_img  = fake_img.transpose(0, 2, 3, 1)
        fake_imgs.append(fake_img)

        if i + 1== max_iter:
            break

    fake_imgs = np.stack(fake_imgs, axis=0)
    if args.display_image:
        show_images(fake_imgs)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f'gen_{i}.jpg')
        cv2.imwrite(save_path, img * 255.0)
    

def main():
    args = parse_args()

    check_params(args)

    print("Init models...")

    G = Generator().cuda()
    D = Discriminator().cuda()
    vgg19 = get_vgg19().cuda()
    
    loss_fn = AnimeGanLoss(args)

    # Create DataLoader
    num_workers = cpu_count()
    data_loader = DataLoader(
        AnimeDataSet(args.data_dir, args.dataset),
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d)

    for e in range(args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)

        for img, anime, anime_gray, anime_smt_gray in bar:

            # To cuda
            img = img.cuda().float()
            anime_gray = anime_gray.cuda().float()
            anime_smt = anime_smt.cuda().float()
            anime_smt_gray = anime_smt_gray.cuda().float()

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

        if e % args.plot_interval == 0:
            save_samples(G, photo_loader, args)


if __name__ == '__main__':
    main()