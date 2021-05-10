import matplotlib.pyplot as plt
from PIL import Image
from util import rgb_to_yuv, rgb_to_yuv_batch
import numpy as np
import torch

image = Image.open("example/10.jpg")
np_img = np.array(image).astype('float32')
np_img = (np_img / 127.5) - 1

img = torch.from_numpy(np_img)


yuv_img = rgb_to_yuv(img, channel_last=True)

# plt.figure()
f, ax = plt.subplots(2, 1)
img = img.numpy()
yuv_img = (yuv_img.numpy() + 1) / 2

print(np.max(yuv_img), np.min(yuv_img))

img = np.expand_dims(img, 0)
print(img.shape)
# ax[0].imshow((np_img + 1) / 2)
# ax[1].imshow(yuv_img)
# plt.show()

import tensorflow as tf
from util import gram

def tf_gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)



print(tf_gram(img))
img = img.transpose((0, 3, 1 ,2))

grammat = gram(torch.from_numpy(img))
print(grammat)
