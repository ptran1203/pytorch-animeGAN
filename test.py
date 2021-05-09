import matplotlib.pyplot as plt
from PIL import Image
from util import rgb_to_yuv, normalize_vgg
import numpy as np
import torch

image = Image.open("example/10.jpg")
np_img = np.array(image).astype('float32')
np_img = (np_img / 127.5) - 1

img = torch.from_numpy(np_img)
img = normalize_vgg(img)

yuv_img = rgb_to_yuv(img, channel_last=True)

# plt.figure()
f, ax = plt.subplots(2, 1)
img = img.numpy()
yuv_img = (yuv_img.numpy() + 1) / 2

print(np.max(yuv_img), np.min(yuv_img))

ax[0].imshow((np_img + 1) / 2)
ax[1].imshow(yuv_img)
plt.show()