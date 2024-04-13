"""
Convert each frame in a anime video to 256 x 256 images
"""
import cv2
import os
import numpy as np
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='/home/ubuntu/Downloads/kimetsu_yaiba.mp4')
    parser.add_argument('--save-path', type=str, default='./script/test_crop')
    parser.add_argument('--max-image', type=int, default=1800)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=256)

    return parser.parse_args()


class VideoConverter:
    def __init__(self, video_path, save_dir, max_image=1600, start=0, end=0, image_size=256):
        os.makedirs(save_dir, exist_ok=True)

        assert os.path.isfile(video_path), f'{video_path} must be a video file'

        self.save_dir = save_dir
        self.video_path = video_path
        self.max_image = max_image
        self.start = start
        self.end = end or None
        self.counter = 0
        self.image_size = image_size

    def crop_and_save(self, image):
        '''
        Crop a large image into smaller images of given size

        @Returns:
            - True if counter reach max_image
        '''

        height, width, _ = image.shape

        # Image can be divided into rows * cols sub-images
        rows = height // self.image_size
        cols = width // self.image_size

        for r in range(rows):
            for c in range(cols):
                start_x = r * self.image_size
                end_x = start_x + self.image_size
                start_y = c * self.image_size
                end_y = start_y + self.image_size
                # print(f'x: [{start_x}:{end_x}], y: [{start_y}:{end_y}]')
                sub_im = image[start_x: end_x, start_y: end_y, :]

                if np.std(sub_im) > 25.0:
                    save_path = os.path.join(self.save_dir, f'{self.counter}.jpg')
                    cv2.imwrite(save_path, sub_im)
                    self.counter += 1

                if self.counter == self.max_image:
                    return True

        return False

    def process(self):
        '''
        Process video
        '''
        video_clip = VideoFileClip(self.video_path)
        if self.start or self.end:
            video_clip = video_clip.subclip(self.start, self.end)

        video_clip = video_clip.set_fps(video_clip.fps // 10)

        total_frames = round(video_clip.fps * video_clip.duration)
        print(f'Processing video {self.video_path}, {total_frames} frames, size: {video_clip.size}')

        for frame in tqdm(video_clip.iter_frames(), total=total_frames):
            # It's better if we resizing before crop to keep the image looks more 'scene'
            h, w, _ = frame.shape
            aspect_ratio = w / h
            ratio = h / (self.image_size * 2)
            w /= ratio
            h = w / aspect_ratio

            frame = cv2.resize(frame[...,::-1], (int(h) , int(w)))
            if self.crop_and_save(frame):
                break

        print(f'Saved {self.counter} images to {self.save_dir}')



if __name__ == '__main__':

    args = parse_args()
    converter = VideoConverter(
        args.video_path,
        args.save_path,
        max_image=args.max_image,
        start=args.start,
        end=args.end,
        image_size=args.image_size
        )

    converter.process()
