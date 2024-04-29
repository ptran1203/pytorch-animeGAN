import os
import argparse
from inference import Predictor
from utils import is_image_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str)
    parser.add_argument('--src', type=str, help='source dir, contain real images')
    parser.add_argument('--device', type=str, default='cuda', help='Device, cuda or cpu')
    parser.add_argument('--out', type=str, default='inference_images', help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    predictor = Predictor(args.weight, args.device)

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        predictor.transform_in_dir(args.src, args.out)
    else:
        save_path = args.out
        if not is_image_file(args.out):
            os.makedirs(args.out, exist_ok=True)
            save_path = os.path.join(args.out, os.path.basename(args.src))
        predictor.transform_file(args.src, save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
