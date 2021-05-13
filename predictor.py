import torch
import cv2
import os
import numpy as np
from modeling.anime_gan import Generator
from util import load_checkpoint, resize_image
from tqdm import tqdm


cuda_available = torch.cuda.is_available()
VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

class Predictor:
    def __init__(self, checkpoint_dir):
        print("Init Generator...")

        self.G = Generator()

        if cuda_available:
            self.G = self.G.cuda()

        load_checkpoint(self.G, checkpoint_dir)
        self.G.eval()

        print("Weight loaded, ready to predict")

    def predict(self, image):
        with torch.no_grad():
            fake = self.G(self.preprocess_images(image))
            fake = fake.detach().cpu().numpy()[0]
            fake = fake.transpose(1, 2, 0)
            return fake

    def predict_batch(self, images):
        with torch.no_grad():
            fake = self.G(self.preprocess_images(images))
            fake = fake.detach().cpu().numpy()
            fake = fake.transpose(0, 2, 3, 1)
            return fake


    def predict_dir(self, img_dir, dest_dir, max_images=0, img_size=(512, 512)):
        '''
        Read all images from img_dir, predict and write the result
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
            anime_img = self.predict(image)
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            anime_img = self.toint16(anime_img)
            cv2.imwrite(os.path.join(dest_dir, f'{fname}_anime.jpg'), anime_img[..., ::-1])

    @staticmethod
    def preprocess_images(images):
        '''
        Preprocess image for inference

        @Arguments:
            - images: np.ndarray

        @Returns
            - images: torch.tensor
        '''
        images = images.astype(np.float32)

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