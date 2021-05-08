import os
import cv2
import numpy as np
import pandas as pd
import torch
# import albumentations as A
from torch.utils.data import Dataset

class AnimeDataSet(Dataset):
    def __init__(self, data_dir, dataset, transform=None):
        """   
        folder structure:
            - {data_dir}
                - photo
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g Hayao
                    smooth
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """
        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        self.data_dir = data_dir
        self.image_files =  {}
        self.photo = 'train_photo'
        self.style = f'{anime_dir}/style'
        self.smooth =  f'{anime_dir}/smooth'

        for opt in [self.photo, self.style, self.smooth]:
            folder = os.path.join(data_dir, opt)
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

    def __len__(self):
        return len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        index_anm = np.random.randint(0, self.len_anime - 1)
        index_smt = np.random.randint(0, self.len_smooth - 1)
        image, _ = self.load_images(index, self.photo)
        anime, anime_gray = self.load_images(index_anm, self.style)
        _, smooth_gray = self.load_images(index_smt, self.smooth)

        return image, anime, anime_gray, smooth_gray


    def load_images(self, index, opt):
        fname = self.image_files[opt][index]
        image = cv2.imread(os.path.join(self.img_dir, fname))[:,:,::-1]

        if opt in {self.style, self.smooth}:
            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
            image_gray = self._transform(image_gray)
            image_gray = image_gray.transpose(2, 0, 1)
            image_gray = torch.tensor(image_gray)
        else:
            h, w, c = image.shape
            image_gray = torch.tensor(np.zeros((c, h, w)))

        image = self._transform(image)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), image_gray

    def _transform(self, img):
        if self.transform is not None:
            img =  self.transform(image=img)['image'].astype(np.float32)

        img = img / 255.0
        return img

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(AnimeDataSet('dataset/Hayao/smooth'), batch_size=2, shuffle=True)

    img, img_gray = iter(anime_loader).next()
    plt.imshow(img[1].numpy().transpose(1, 2, 0))
    plt.show()
