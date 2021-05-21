import matplotlib.pyplot as plt
from PIL import Image
from util import rgb_to_yuv
import numpy as np
import torch
from util import gram, DefaultArgs

def rgb_to_yuv_test(image):
    import tensorflow as tf

    image = (image + 1.0)/2.0
    return tf.image.rgb_to_yuv(image)


image = Image.open("example/10.jpg")
image = image.resize((256, 256))
np_img = np.array(image).astype('float32')
np_img = (np_img / 127.5) - 1

img = torch.from_numpy(np_img)

yuv_img = rgb_to_yuv(img.permute(2, 0, 1).type(torch.float32).unsqueeze(0))[0]
# yuv_test = rgb_to_yuv_test(img)

print('yuv_img output', yuv_img.shape)
# plt.figure()

img = img.numpy()

img = np.expand_dims(img, 0)
# print(yuv_img.shape, yuv_test.shape)

f, ax = plt.subplots(4, 1)
# yuv_test = yuv_test.numpy()

kernel = np.array([
    [0, 1, -1, 0],
    [1, 3, -3, -1],
    [1, 3, -3, -1],
    [0, 1, -1, 0]
])



yuv_img = yuv_img.numpy()
ax[0].imshow((np_img + 1) / 2)
ax[1].imshow(yuv_img)

plt.show()


from modeling.anime_gan import Generator, Discriminator

D = Discriminator(DefaultArgs())
G = Generator()

img = img.transpose(0, 3, 1, 2)
fake = G(torch.from_numpy(img))
pred = D(torch.from_numpy(img))

print(G)

print(f'G: {fake.shape}, D: {pred.shape}')
