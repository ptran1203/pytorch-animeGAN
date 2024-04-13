import os
import cv2
import numpy as np
import pandas as pd
import torch
from glob import glob
from torch.utils.data import Dataset
from utils import normalize_input, compute_data_mean

CACHE_DIR = 'tmp'

class AnimeDataSet(Dataset):
    def __init__(
        self,
        anime_image_dir,
        real_image_dir,
        debug_samples=0,
        cache=False,
        transform=None
    ):
        """   
        folder structure:
        - {anime_image_dir}  # E.g Hayao
            smooth
                1.jpg, ..., n.jpg
            style
                1.jpg, ..., n.jpg
        """
        self.cache = cache
        self.mean = compute_data_mean(os.path.join(anime_image_dir, 'style'))
        print(f'Mean(B, G, R) of {anime_image_dir} are {self.mean}')

        self.debug_samples = debug_samples
        self.image_files =  {}
        self.photo = 'train_photo'
        self.style = 'style'
        self.smooth =  'smooth'
        self.dummy = torch.zeros(3, 256, 256)

        for dir, opt in [
            (real_image_dir, self.photo),
            (os.path.join(anime_image_dir, self.style), self.style),
            (os.path.join(anime_image_dir, self.smooth), self.smooth)
        ]:
            self.image_files[opt] = glob(os.path.join(dir, "*.*"))

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return {
            "image": image,
            "anime": anime,
            "anime_gray": anime_gray,
            "smooth_gray": smooth_gray
        }

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)[:,:,::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)[:,:,::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(AnimeDataSet('dataset/Hayao/smooth'), batch_size=2, shuffle=True)

    img, img_gray = iter(anime_loader).next()
    plt.imshow(img[1].numpy().transpose(1, 2, 0))
    plt.show()
