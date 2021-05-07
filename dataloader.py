import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        
        image_dir has the format: {dir}/{photo|anime|anime_smooth}/
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f'Folder {image_dir} does not exist')

        self.img_dir = image_dir
        self.image_files = os.listdir(self.img_dir)
        if not self.image_files:
            raise Exception(f'{self.img_dir} has no files')

        self.transform = transform
        self.is_anime = 'anime' in image_dir or 'style' in image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fname = self.image_files[index]
        image = cv2.imread(os.path.join(self.img_dir, fname))[:,:,::-1]

        if self.is_anime:
            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_gray = self._transform(image_gray)
            image_gray = image_gray.transpose(2, 0, 1)
            image_gray = torch.tensor(image_gray)
        else:
            image_gray = None

        return torch.tensor(image), image_gray

    def _transform(self, img):
        if self.transform is not None:
            img =  self.transform(image=img)['image'].astype(np.float32)

        return img
