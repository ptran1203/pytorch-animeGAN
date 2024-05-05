import torch
from utils import compute_data_mean
from models.anime_gan_v2 import GeneratorV2
from models.anime_gan import GeneratorV1, Discriminator
# print(compute_data_mean('dataset/Hayao/style'))
from color_transfer import color_transfer_pytorch
image = torch.rand(2, 3, 256, 256)

gray = to_gray_scale(image)

print(gray.shape, gray[0, :, 1, 1])

# G = GeneratorV2()
# D = Discriminator(num_layers=3, norm_type="layer")


# image = torch.rand(2, 3, 256, 256)

# out = G(image)

# d = D(image)

# print(out.shape, d.shape)

# import cv2
# import numpy as np

# # 
# img1 = cv2.imread("/u02/phatth1/dataset/Hayao/style/6.jpg")[:, :, ::-1]
# img2 = cv2.imread("/u02/phatth1/dataset/Hayao/style/6.jpg")[:, :, ::-1]
# image = np.stack([img1, img2]).astype('float32') / 255.0
# print(image.shape)
# image = image.transpose(0, 3, 1, 2)
# image = torch.tensor(image)

# image = image.permute(0, 2, 3, 1).double()

# _rgb_to_yuv_kernel = torch.tensor([
#             [0.299, 0.587, 0.114],
#             [-0.14714119, -0.28886916, 0.43601035],
#             [0.61497538, -0.51496512, -0.10001026],
#         ]).double()

# # yuv_img = torch.tensordot(
# #     image,
# #     _rgb_to_yuv_kernel,
# #     dims=([-1], [0]))
# print(image.shape, _rgb_to_yuv_kernel)
# yuv_img = image @ _rgb_to_yuv_kernel.T

# yuv_img = yuv_img * 255.0
# yuv_img = yuv_img.numpy().astype(np.uint8)

# cv2.imwrite("test1.jpg", yuv_img[0])
# cv2.imwrite("test2.jpg", yuv_img[1])

# # print(image.ndim)


# yuv_from_rgb = np.array(
#     [
#         [0.299, 0.587, 0.114],
#         [-0.14714119, -0.28886916, 0.43601035],
#         [0.61497538, -0.51496512, -0.10001026],
#     ]
# )

# img1 = img1.astype(float) / 255.0
# yuv_np = img1 @ yuv_from_rgb.T.astype(img1.dtype)
# yuv_np = (yuv_np * 255).astype(np.uint8)

# print((yuv_np - yuv_img[0]).mean())

# print(yuv_from_rgb.dtype)
