import torch
import cv2


_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()

if torch.cuda.is_available():
    _rgb_to_yuv_kernel = _rgb_to_yuv_kernel.cuda()


def gram(input):
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image):
    '''
    https://en.wikipedia.org/wiki/YUV

    output: Image of shape (H, W, C) (channel last)
    '''
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(
        image,
        _rgb_to_yuv_kernel,
        dims=([image.ndim - 3], [0]))

    return yuv_img


def resize_image(img, size):
    h, w = img.shape[:2]

    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y

    img = cv2.resize(img, (w, h))
    return img


def normalize_input(images):
    return images / 127.5 - 1.0


def denormalize_input(images, dtype=None):
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, np.ndarray):
            images = images.astype(dtype)
        else:
            # Torch
            images = images.type(dtype)

    return images
