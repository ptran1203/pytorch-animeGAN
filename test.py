import matplotlib.pyplot as plt
from PIL import Image
from util import rgb_to_yuv
import numpy as np
import torch

image = Image.open("example/10.jpg")
np_img = np.array(image).astype('float32')

yuv_img = rgb_to_yuv(torch.from_numpy(np_img), channel_last=True)

y, u, v = torch.split(yuv_img, dim=0)

print(y.shape, u.shape, v.shape)

# plt.figure()
f, ax = plt.subplots(2, 1)
ax[0].imshow(image)
ax[1].imshow(yuv_img)
plt.show()