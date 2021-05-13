import argparse
from inference.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='/content/images', help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    predictor = Predictor(args.checkpoint)
    predictor.predict_dir(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)
