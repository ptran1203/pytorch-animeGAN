import torch
import argparse
import os
from models.anime_gan import GeneratorV1
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan_v3 import GeneratorV3
from models.anime_gan import Discriminator
from datasets import AnimeDataSet
from utils.common import load_checkpoint
from trainer import Trainer
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_image_dir', type=str, default='dataset/train_photo')
    parser.add_argument('--anime_image_dir', type=str, default='dataset/Hayao')
    parser.add_argument('--test_image_dir', type=str, default='dataset/test/HR_photo')
    parser.add_argument('--model', type=str, default='v1', help="AnimeGAN version, can be {'v1', 'v2', 'v3'}")
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--init_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--exp_dir', type=str, default='runs', help="Experiment directory")
    parser.add_argument('--gan_loss', type=str, default='lsgan', help='lsgan / hinge / bce')
    parser.add_argument('--resume', action='store_true', help="Continue from current dir")
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
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--resize_method', type=str, default="crop",
                        help="Resize image method if origin photo larger than imgsz")
    parser.add_argument('--lr_g', type=float, default=2e-5)
    parser.add_argument('--lr_d', type=float, default=4e-5)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--wadvg', type=float, default=300.0, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', type=float, default=300.0, help='Adversarial loss weight for D')
    # Loss weight VGG19
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight') #  1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    parser.add_argument('--wgra', type=float, default=5.0, help='Gram loss weight') # 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    parser.add_argument('--wcol', type=float, default=30.0, help='Color loss weight') # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--wtvar', type=float, default=1.0, help='Total variation loss') # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    parser.add_argument('--d_layers', type=int, default=2, help='Discriminator conv layers')
    parser.add_argument('--d_noise', action='store_true')

    # DDP
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--world-size", default=2, type=int)

    return parser.parse_args()



def check_params(args):
    # dataset/Hayao + dataset/train_photo -> train_photo_Hayao
    args.dataset = f"{os.path.basename(args.real_image_dir)}_{os.path.basename(args.anime_image_dir)}"
    assert args.gan_loss in {'lsgan', 'hinge', 'bce'}, f'{args.gan_loss} is not supported'


def main(args, logger):
    check_params(args)

    if not torch.cuda.is_available():
        logger.info("CUDA not found, use CPU")
        # Just for debugging purpose, set to minimum config
        # to avoid 🔥 the computer...
        args.device = 'cpu'
        args.debug_samples = 10
        args.batch_size = 2
    else:
        logger.info(f"Use GPU: {torch.cuda.get_device_name(0)}")

    norm_type = "instance"
    if args.model == 'v1':
        G = GeneratorV1(args.dataset)
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

    trainer = Trainer(
        generator=G,
        discriminator=D,
        config=args,
        logger=logger,
    )

    if args.resume_G_init.lower() != 'false':
        start_e_init = load_checkpoint(G, args.resume_G_init)
        if args.local_rank == 0:
            logger.info(f"G content weight loaded from {args.resume_G_init}")
    elif args.resume_G.lower() != 'false' and args.resume_D.lower() != 'false':
        # You should provide both
        try:
            start_e = load_checkpoint(G, args.resume_G)
            if args.local_rank == 0:
                logger.info(f"G weight loaded from {args.resume_G}")
            load_checkpoint(D, args.resume_D)
            if args.local_rank == 0:
                logger.info(f"D weight loaded from {args.resume_D}")
            # If loaded both weight, turn off init G phrase
            args.init_epochs = 0

        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)
    elif args.resume:
        # Try to load from working dir
        logger.info(f"Loading weight from {trainer.checkpoint_path_G}")
        start_e = load_checkpoint(G, trainer.checkpoint_path_G)
        logger.info(f"Loading weight from {trainer.checkpoint_path_D}")
        load_checkpoint(D, trainer.checkpoint_path_D)
        args.init_epochs = 0
    
    dataset = AnimeDataSet(
        args.anime_image_dir,
        args.real_image_dir,
        args.debug_samples,
        args.cache,
        imgsz=args.imgsz,
        resize_method=args.resize_method,
    )
    if args.local_rank == 0:
        logger.info(f"Start from epoch {start_e}, {start_e_init}")
    trainer.train(dataset, start_e, start_e_init)

if __name__ == '__main__':
    args = parse_args()
    real_name = os.path.basename(args.real_image_dir)
    anime_name = os.path.basename(args.anime_image_dir)
    args.exp_dir = f"{args.exp_dir}_{real_name}_{anime_name}_{args.imgsz}_{args.wadvg}"

    os.makedirs(args.exp_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.exp_dir, "train.log"))

    if args.local_rank == 0:
        logger.info("# ==== Train Config ==== #")
        for arg in vars(args):
            logger.info(f"{arg} {getattr(args, arg)}")
        logger.info("==========================")

    main(args, logger)
