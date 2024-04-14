import torch
import argparse
import os
import cv2
import numpy as np
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from models.anime_gan import Generator
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan_v3 import GeneratorV3
from models.anime_gan import Discriminator
from losses import AnimeGanLoss
from losses import LossSummary
from utils.common import load_checkpoint
from dataset import AnimeDataSet
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_image_dir', type=str, default='dataset/train_photo')
    parser.add_argument('--anime_image_dir', type=str, default='dataset/Hayao')
    parser.add_argument('--model', type=str, default='v1', help="AnimeGAN version, can be {'v1', 'v2', 'v3'}")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--exp_dir', type=str, default='runs', help="Experiment directory")
    parser.add_argument('--gan_loss', type=str, default='lsgan', help='lsgan / hinge / bce')
    parser.add_argument('--resume_G_init', type=str, default='False')
    parser.add_argument('--resume_G', type=str, default='False')
    parser.add_argument('--resume_D', type=str, default='False')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--cache', action='store_true', help="Turn on disk cache")
    parser.add_argument('--amp', action='store_true', help="Turn on Automatic Mixed Precision")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--debug_samples', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=4e-4)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--wadvg', type=float, default=10.0, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', type=float, default=10.0, help='Adversarial loss weight for D')
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=3.0, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=30.0, help='Color loss weight')
    parser.add_argument('--d_layers', type=int, default=3, help='Discriminator conv layers')
    parser.add_argument('--d_noise', action='store_true')

    return parser.parse_args()



def check_params(args):
    # dataset/Hayao -> Hayao
    args.dataset = os.path.basename(args.anime_image_dir)
    assert args.gan_loss in {'lsgan', 'hinge', 'bce'}, f'{args.gan_loss} is not supported'


def main(args):
    check_params(args)

    if not torch.cuda.is_available():
        print("CUDA not found, use CPU")
        # Just for debugging purpose, set to minimum config
        # to avoid 🔥 the computer...
        args.device = 'cpu'
        args.debug_samples = 10
        args.batch_size = 2
    else:
        print(f"Use GPU: {torch.cuda.get_device_name(0)}")
    print("Init models...")

    norm_type = "instance"
    if args.model == 'v1':
        G = Generator(args.dataset)
    elif args.model == 'v2':
        G = GeneratorV2(args.dataset)
        norm_type = "layer"
    elif args.model == 'v3':
        G = GeneratorV3(args.dataset)

    D = Discriminator(
        args.dataset,
        num_layers=args.d_layers,
        use_sn=args.use_sn,
        norm_type=norm_type,
    )

    start_e = 0
    start_e_init = 0

    if args.resume_G_init.lower() != 'false':
        start_e_init = load_checkpoint(G, args.resume_G_init)
        print(f"G content weight loaded from {args.resume_G_init}")
    elif args.resume_G.lower() != 'false' and args.resume_D.lower() != 'false':
        # You should provide both
        try:
            start_e = load_checkpoint(G, args.resume_G)
            print(f"G weight loaded from {args.resume_G}")
            load_checkpoint(D, args.checkpoint_dir)
            print(f"D weight loaded from {args.resume_D}")
            # If loaded both weight, turn off init G phrase
            args.init_epochs = 0

        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)

    trainer = Trainer(
        generator=G,
        discriminator=D,
        config=args,
    )
    dataset = AnimeDataSet(
        args.anime_image_dir,
        args.real_image_dir,
        args.debug_samples,
        args.cache,
    )
    trainer.train(dataset, start_e, start_e_init)

if __name__ == '__main__':
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)
