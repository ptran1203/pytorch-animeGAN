import matplotlib.pyplot as plt
from PIL import Image
from util import rgb_to_yuv, rgb_to_yuv_batch
import numpy as np
import torch
from util import gram, DefaultArgs


image = Image.open("example/10.jpg")
image = image.resize((256, 256))
np_img = np.array(image).astype('float32')
np_img = (np_img / 127.5) - 1

img = torch.from_numpy(np_img)

yuv_img = rgb_to_yuv_batch(img.permute(2, 0, 1).type(torch.float32).unsqueeze(0))[0]
print('yuv_img output', yuv_img.shape)
# plt.figure()
f, ax = plt.subplots(2, 1)

img = img.numpy()
yuv_img = (yuv_img.numpy() + 1) / 2

img = np.expand_dims(img, 0)
print(yuv_img.shape)

# ax[0].imshow((np_img + 1) / 2)
# ax[1].imshow(yuv_img)
# plt.show()


# img = img.transpose((0, 3, 1 ,2))
# grammat = gram(torch.from_numpy(img))
# print(grammat)


from modeling.anime_gan import Generator, Discriminator

D = Discriminator(DefaultArgs())
G = Generator()

img = img.transpose(0, 3, 1, 2)
fake = G(torch.from_numpy(img))
pred = D(torch.from_numpy(img))

print(f'G: {fake.shape}, D: {pred.shape}')