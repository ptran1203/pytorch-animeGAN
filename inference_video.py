import argparse
from inference import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='Path to input video')
    parser.add_argument('--dest', type=str, default='/content/images', help='Path to save new video')
    parser.add_argument('--batch-size', type=int, default=4)

    return parser.parse_args()


def main(args):
    Transformer(args.checkpoint).transform_video(args.src, args.dest, args.batch_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)
