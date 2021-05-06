import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, csv, mode='train', transform=None, image_dir=''):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f'Folder {image_dir} does not exist')

        self.img_dir = image_dir
        self.image_files = os.listdir(self.img_dir)
        if not self.image_files:
            raise Exception(f'{self.img_dir} has no files')

        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        fname = self.image_files[index]
        image = cv2.imread(os.path.join(self.img_dir, fname))[:,:,::-1]

        # transform
        image = self.transform(image=image)['image'].astype(np.float32)

        image = image.transpose(2, 0, 1)

        return torch.tensor(image)