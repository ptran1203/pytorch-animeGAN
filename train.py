import torch
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator

g = Generator()
d = Discriminator()

img = torch.randn((1, 3, 128, 128))

fake = d(img)

print(fake.shape)