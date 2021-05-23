import torch
import cv2
import os
import numpy as np
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from modeling.anime_gan import Generator
from util import load_checkpoint, resize_image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


cuda_available = torch.cuda.is_available()
VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

class Transformer:
    def __init__(self, checkpoint_dir, add_mean=False):
        print("Init Generator...")

        self.G = Generator()
        self.add_mean = add_mean

        if cuda_available:
            self.G = self.G.cuda()

        load_checkpoint(self.G, checkpoint_dir)
        self.G.eval()

        print("Weight loaded, ready to predict")

    def transform(self, image):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        with torch.no_grad():
            fake = self.G(self.preprocess_images(image))
            fake = fake.detach().cpu().numpy()
            # Channel last
            fake = fake.transpose(0, 2, 3, 1)
            return fake

    def transform_in_dir(self, img_dir, dest_dir, max_images=0, img_size=(512, 512)):
        '''
        Read all images from img_dir, transform and write the result
        to dest_dir

        '''
        os.makedirs(dest_dir, exist_ok=True)

        files = os.listdir(img_dir)
        files = [f for f in files if self.is_valid_file(f)]
        print(f'Found {len(files)} images in {img_dir}')

        if max_images:
            files = files[:max_images]

        for fname in tqdm(files):
            image = cv2.imread(os.path.join(img_dir, fname))[:,:,::-1]
            image = resize_image(image, img_size)
            anime_img = self.transform(image)
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            anime_img = self.toint16(anime_img)
            cv2.imwrite(os.path.join(dest_dir, f'{fname}_anime.jpg'), anime_img[..., ::-1])


    def transfrom_video(self, input_path, output_path, batch_size=4):
        '''
        Transform a video to animation version
        https://github.com/lengstrom/fast-style-transfer/blob/master/evaluate.py#L21
        '''
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        output_dir = "/".join(output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)

        print(f'Transfroming video {input_path}')

        def transform_and_write(frames, count, writer):
            anime_images = self.toint8(self.transform(frames))
            for i in range(0, count):
                img = np.clip(anime_images[i], 0, 255)
                writer.write_frame(img)

        video_clip = VideoFileClip(input_path, audio=False)
        video_writer = ffmpeg_writer.FFMPEG_VideoWriter(output_path, video_clip.size, video_clip.fps, codec="libx264",
                                                        preset="medium", bitrate="2000k",
                                                        audiofile=input_path, threads=None,
                                                        ffmpeg_params=None)

        frame_count = 0
        frames = np.zeros(batch_size, dtype=np.float32)

        for frame in video_clip.iter_frames():
            frames[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                transform_and_write(frames, frame_count, video_writer)
                frame_count = 0

        # The last frames
        if frame_count != 0:
            transform_and_write(frames, frame_count, video_writer)

        print(f'Animation video saved to {output_path}')
        video_writer.close()

    def preprocess_images(self, images):
        '''
        Preprocess image for inference

        @Arguments:
            - images: np.ndarray

        @Returns
            - images: torch.tensor
        '''
        images = images.astype(np.float32)
        if self.add_mean:
            images[:,:, 0] += -4.4661
            images[:,:, 1] += -8.6698
            images[:,:, 2] += 13.1360

        # Normalize to [-1, 1]
        images = (images / 127.5) - 1.0
        images = torch.from_numpy(images)

        if cuda_available:
            images = images.cuda()

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # channel first
        images = images.permute(0, 3, 1, 2)

        return images


    @staticmethod
    def is_valid_file(fname):
        ext = fname.split('.')[-1]
        return ext in VALID_FORMATS

    @staticmethod
    def toint16(img):
        img = img * 127.5 + 127.5
        img = img.astype(np.int16)
        return img

    @staticmethod
    def toint8(img):
        img = img * 127.5 + 127.5
        img = img.astype(np.uint8)
        return img