import torch
# from utils import compute_data_mean
from models.anime_gan_v2 import GeneratorV2
# from models.anime_gan import GeneratorV1, Discriminator
# print(compute_data_mean('dataset/Hayao/style'))


model = GeneratorV2()

inp = torch.rand(2, 3, 256, 256)

out = model(inp)

print(out.shape)