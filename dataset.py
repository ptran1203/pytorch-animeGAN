import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
from tqdm.auto import tqdm
from glob import glob
from torch.utils.data import Dataset
from utils import normalize_input, compute_data_mean, fast_numpyio

CACHE_DIR = '/tmp/animegan_cache'

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
        self.cache_files = {}
        for dir, opt in [
            (real_image_dir, self.photo),
            (os.path.join(anime_image_dir, self.style), self.style),
            (os.path.join(anime_image_dir, self.smooth), self.smooth)
        ]:
            self.image_files[opt] = glob(os.path.join(dir, "*.*"))
            self.cache_files[opt] = [False] * len(self.image_files[opt])

        self.transform = transform
        self.cache_data()

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
        anm_idx = random.randint(0, self.len_anime - 1)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return {
            "image": torch.tensor(image),
            "anime": torch.tensor(anime),
            "anime_gray": torch.tensor(anime_gray),
            "smooth_gray": torch.tensor(smooth_gray)
        }

    def cache_data(self):
        if not self.cache:
            return
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Caching image to npy for faster dataloader
        print("Caching data..")
        cache_nbytes = 0
        for opt, image_files in self.image_files.items():
            cache_sub_sir = os.path.join(CACHE_DIR, opt)
            os.makedirs(cache_sub_sir, exist_ok=True)
            for index, img_file in enumerate(tqdm(image_files)):
                save_path = os.path.join(cache_sub_sir, f"{index}.npy")
                if os.path.exists(save_path):
                    continue  # Cache exist.
                if opt == self.photo:
                    image = self.load_photo(index)
                    cache_nbytes += image.nbytes
                    fast_numpyio.save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.smooth:
                    cache_nbytes += image.nbytes
                    image = self.load_anime_smooth(index)
                    np.save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.style:
                    image, image_gray = self.load_anime(index)
                    cache_nbytes = cache_nbytes + image.nbytes + image_gray.nbytes
                    fast_numpyio.save(save_path, image)
                    save_path_gray = os.path.join(cache_sub_sir, f"{index}_gray.npy")
                    fast_numpyio.save(save_path_gray, image_gray)
                    self.cache_files[opt][index] = (save_path, save_path_gray)
                else:
                    raise ValueError(opt)
        print(f"Cache saved to {CACHE_DIR}, size={cache_nbytes/1e9} Gb")

    def load_photo(self, index) -> np.ndarray:
        if self.cache_files[self.photo][index]:
            fpath = self.cache_files[self.photo][index]
            image = fast_numpyio.load(fpath)
        else:
            fpath = self.image_files[self.photo][index]
            image = cv2.imread(fpath)[:,:,::-1]
            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
        return image

    def load_anime(self, index) -> np.ndarray:
        if self.cache_files[self.style][index]:
            fpath, fpath_gray = self.cache_files[self.style][index]
            image = fast_numpyio.load(fpath)
            image_gray = fast_numpyio.load(fpath_gray)
        else:
            fpath = self.image_files[self.style][index]
            image = cv2.imread(fpath)[:,:,::-1]

            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
            image_gray = self._transform(image_gray, addmean=False)
            image_gray = image_gray.transpose(2, 0, 1)

            image = self._transform(image, addmean=True)
            image = image.transpose(2, 0, 1)

            image = np.ascontiguousarray(image)
            image_gray = np.ascontiguousarray(image_gray)

        return image, image_gray

    def load_anime_smooth(self, index) -> np.ndarray:
        if self.cache_files[self.smooth][index]:
            fpath = self.cache_files[self.smooth][index]
            image = fast_numpyio.load(fpath)
        else:
            fpath = self.image_files[self.smooth][index]
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            image = np.stack([image, image, image], axis=-1)
            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
        return image

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
