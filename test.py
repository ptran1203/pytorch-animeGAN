import torch
from utils import compute_data_mean
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan import GeneratorV1, Discriminator
# print(compute_data_mean('dataset/Hayao/style'))

    


G = GeneratorV2()
D = Discriminator(num_layers=3, norm_type="layer")


image = torch.rand(2, 3, 256, 256)

out = G(image)

d = D(image)

print(out.shape, d.shape)
