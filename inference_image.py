import os
import argparse
from inference import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='/content/images', help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    transformer = Transformer(args.checkpoint)

    if os.path.isfile(args.src):
        transformer.transform_file(args.src, args.dest)
    else:
        transformer.transform_in_dir(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)
