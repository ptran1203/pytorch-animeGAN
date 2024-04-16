import argparse
from inference import Predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='Path to input video')
    parser.add_argument('--dest', type=str, default='/content/images', help='Path to save new video')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--start', type=int, default=0, help='Start time of video (second)')
    parser.add_argument('--end', type=int, default=0, help='End time of video (second), 0 if not set')

    return parser.parse_args()


def main(args):
    Predictor(args.checkpoint).transform_video(args.src, args.dest,
                                                 args.batch_size,
                                                 start=args.start,
                                                 end=args.end)

if __name__ == '__main__':
    args = parse_args()
    main(args)
