import os
import time
import shutil

import torch
import cv2
import numpy as np

from models.anime_gan import GeneratorV1
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan_v3 import GeneratorV3
from utils.common import load_checkpoint, RELEASED_WEIGHTS
from utils.image_processing import resize_image, normalize_input, denormalize_input
from utils import read_image, is_image_file, is_video_file
from tqdm import tqdm
from color_transfer import color_transfer_pytorch


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    ffmpeg_writer = None
    VideoFileClip = None


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - started_at
        print(f"Processed in {elapsed:.3f}s")
        return result
    return wrap


def auto_load_weight(weight, version=None, map_location=None):
    """Auto load Generator version from weight."""
    weight_name = os.path.basename(weight).lower()
    if version is not None:
        version = version.lower()
        assert version in {"v1", "v2", "v3"}, f"Version {version} does not exist"
        # If version is provided, use it.
        cls = {
            "v1": GeneratorV1,
            "v2": GeneratorV2,
            "v3": GeneratorV3
        }[version]
    else:
        # Try to get class by name of weight file    
        # For convenenice, weight should start with classname
        # e.g: Generatorv2_{anything}.pt
        if weight_name in RELEASED_WEIGHTS:
            version = RELEASED_WEIGHTS[weight_name][0]
            return auto_load_weight(weight, version=version, map_location=map_location)

        elif weight_name.startswith("generatorv2"):
            cls = GeneratorV2
        elif weight_name.startswith("generatorv3"):
            cls = GeneratorV3
        elif weight_name.startswith("generator"):
            cls = GeneratorV1
        else:
            raise ValueError((f"Can not get Model from {weight_name}, "
                               "you might need to explicitly specify version"))
    model = cls()
    load_checkpoint(model, weight, strip_optimizer=True, map_location=map_location)
    model.eval()
    return model


class Predictor:
    """
    Generic class for transfering Image to anime like image.
    """
    def __init__(
        self,
        weight='hayao',
        device='cuda',
        amp=True,
        retain_color=False,
        imgsz=None,
    ):
        if not torch.cuda.is_available():
            device = 'cpu'
            # Amp not working on cpu
            amp = False
            print("Use CPU device")
        else:
            print(f"Use GPU {torch.cuda.get_device_name()}")
        
        self.imgsz = imgsz
        self.retain_color = retain_color
        self.amp = amp  # Automatic Mixed Precision
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.device = torch.device(device)
        self.G = auto_load_weight(weight, map_location=device)
        self.G.to(self.device)

    def transform_and_show(
        self,
        image_path,
        figsize=(18, 10),
        save_path=None
    ):
        image = resize_image(read_image(image_path))
        anime_img = self.transform(image)
        anime_img = anime_img.astype('uint8')

        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1, 2, 1)
        # plt.title("Input")
        plt.imshow(image)
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        # plt.title("Anime style")
        plt.imshow(anime_img[0])
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def transform(self, image, denorm=True):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        with torch.no_grad():
            image = self.preprocess_images(image)
            # image = image.to(self.device)
            # with autocast(self.device_type, enabled=self.amp):
                # print(image.dtype, self.G)
            fake = self.G(image)
            # Transfer color of fake image look similiar color as image
            if self.retain_color:
                fake = color_transfer_pytorch(fake, image)
                fake = (fake / 0.5) - 1.0  # remap to [-1. 1]
            fake = fake.detach().cpu().numpy()
            # Channel last
            fake = fake.transpose(0, 2, 3, 1)

            if denorm:
                fake = denormalize_input(fake, dtype=np.uint8)
            return fake

    def read_and_resize(self, path, max_size=1536):
        image = read_image(path)
        _, ext = os.path.splitext(path)
        h, w = image.shape[:2]
        if self.imgsz is not None:
            image = resize_image(image, width=self.imgsz)
        elif max(h, w) > max_size:
            print(f"Image {os.path.basename(path)} is too big ({h}x{w}), resize to max size {max_size}")
            image = resize_image(
                image,
                width=max_size if w > h else None,
                height=max_size if w < h else None,
            )
            cv2.imwrite(path.replace(ext, ".jpg"), image[:,:,::-1])
        else:
            image = resize_image(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = np.stack([image, image, image], -1)
        # cv2.imwrite(path.replace(ext, ".jpg"), image[:,:,::-1])
        return image

    @profile
    def transform_file(self, file_path, save_path):
        if not is_image_file(save_path):
            raise ValueError(f"{save_path} is not valid")

        image = self.read_and_resize(file_path)
        anime_img = self.transform(image)[0]
        cv2.imwrite(save_path, anime_img[..., ::-1])
        print(f"Anime image saved to {save_path}")

    @profile
    def transform_gif(self, file_path, save_path, batch_size=4):
        import imageio

        def _preprocess_gif(img):
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return resize_image(img)

        images = imageio.mimread(file_path)
        images = np.stack([
            _preprocess_gif(img)
            for img in images
        ])

        print(images.shape)

        anime_gif = np.zeros_like(images)

        for i in tqdm(range(0, len(images), batch_size)):
            end = i + batch_size
            anime_gif[i: end] = self.transform(
                images[i: end]
            )

        if end < len(images) - 1:
            # transform last frame
            print("LAST", images[end: ].shape)
            anime_gif[end:] = self.transform(images[end:])

        print(anime_gif.shape)
        imageio.mimsave(
            save_path,
            anime_gif,
            
        )
        print(f"Anime image saved to {save_path}")

    @profile
    def transform_in_dir(self, img_dir, dest_dir, max_images=0, img_size=(512, 512)):
        '''
        Read all images from img_dir, transform and write the result
        to dest_dir

        '''
        os.makedirs(dest_dir, exist_ok=True)

        files = os.listdir(img_dir)
        files = [f for f in files if is_image_file(f)]
        print(f'Found {len(files)} images in {img_dir}')

        if max_images:
            files = files[:max_images]

        bar = tqdm(files)
        for fname in bar:
            path = os.path.join(img_dir, fname)
            image = self.read_and_resize(path)
            anime_img = self.transform(image)[0]
            # anime_img = resize_image(anime_img, width=320)
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            cv2.imwrite(os.path.join(dest_dir, f'{fname}.jpg'), anime_img[..., ::-1])
            bar.set_description(f"{fname} {image.shape}")

    def transform_video(self, input_path, output_path, batch_size=4, start=0, end=0):
        '''
        Transform a video to animation version
        https://github.com/lengstrom/fast-style-transfer/blob/master/evaluate.py#L21
        '''
        if VideoFileClip is None:
            raise ImportError("moviepy is not installed, please install with `pip install moviepy>=1.0.3`")
        # Force to None
        end = end or None

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        is_gg_drive = '/drive/' in output_path
        temp_file = ''

        if is_gg_drive:
            # Writing directly into google drive can be inefficient
            temp_file = f'tmp_anime.{output_path.split(".")[-1]}'

        def transform_and_write(frames, count, writer):
            anime_images = self.transform(frames)
            for i in range(0, count):
                img = np.clip(anime_images[i], 0, 255)
                writer.write_frame(img)

        video_clip = VideoFileClip(input_path, audio=False)
        if start or end:
            video_clip = video_clip.subclip(start, end)

        video_writer = ffmpeg_writer.FFMPEG_VideoWriter(
            temp_file or output_path,
            video_clip.size, video_clip.fps,
            codec="libx264",
            # preset="medium", bitrate="2000k",
            ffmpeg_params=None)

        total_frames = round(video_clip.fps * video_clip.duration)
        print(f'Transfroming video {input_path}, {total_frames} frames, size: {video_clip.size}')

        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        frame_count = 0
        frames = np.zeros(batch_shape, dtype=np.float32)
        for frame in tqdm(video_clip.iter_frames(), total=total_frames):
            try:
                frames[frame_count] = frame
                frame_count += 1
                if frame_count == batch_size:
                    transform_and_write(frames, frame_count, video_writer)
                    frame_count = 0
            except Exception as e:
                print(e)
                break

        # The last frames
        if frame_count != 0:
            transform_and_write(frames, frame_count, video_writer)

        if temp_file:
            # move to output path
            shutil.move(temp_file, output_path)

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

        # Normalize to [-1, 1]
        images = normalize_input(images)
        images = torch.from_numpy(images)

        images = images.to(self.device)

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # channel first
        images = images.permute(0, 3, 1, 2)

        return images


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight',
        type=str,
        default="hayao:v2",
        help=f'Model weight, can be path or pretrained {tuple(RELEASED_WEIGHTS.keys())}'
    )
    parser.add_argument('--src', type=str, help='Source, can be directory contains images, image file or video file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device, cuda or cpu')
    parser.add_argument('--imgsz', type=int, default=None, help='Resize image to specified size if provided')
    parser.add_argument('--out', type=str, default='inference_images', help='Output, can be directory or file')
    parser.add_argument(
        '--retain-color',
        action='store_true',
        help='If provided the generated image will retain original color of input image')
    # Video params
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size when inference video')
    parser.add_argument('--start', type=int, default=0, help='Start time of video (second)')
    parser.add_argument('--end', type=int, default=0, help='End time of video (second), 0 if not set')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    predictor = Predictor(
        args.weight,
        args.device,
        retain_color=args.retain_color,
        imgsz=args.imgsz,
    )

    if not os.path.exists(args.src):
        raise FileNotFoundError(args.src)

    if is_video_file(args.src):
        predictor.transform_video(
            args.src,
            args.out,
            args.batch_size,
            start=args.start,
            end=args.end
        )
    elif os.path.isdir(args.src):
        predictor.transform_in_dir(args.src, args.out)
    elif os.path.isfile(args.src):
        save_path = args.out
        if not is_image_file(args.out):
            os.makedirs(args.out, exist_ok=True)
            save_path = os.path.join(args.out, os.path.basename(args.src))

        if args.src.endswith('.gif'):
            # GIF file
            predictor.transform_gif(args.src, save_path, args.batch_size)
        else:
            predictor.transform_file(args.src, save_path)
    else:
        raise NotImplementedError(f"{args.src} is not supported")
