import torch
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator
from dataloader import DataLoader

def main():
    G = Generator()
    D = Discriminator()

    photo_loader = DataLoader()
    anime_loader = DataLoader()
    anime_smooth_loader = DataLoader()


if __name__ == '__main__':
    main()