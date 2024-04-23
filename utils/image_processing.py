import torch
import cv2
import os
import numpy as np
from tqdm import tqdm


def gram(input):
    """
    Calculate Gram Matrix

    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = input.size()

    x = input.contiguous().view(b * c, w * h)

    # x = x / 2

    # Work around, torch.mm would generate some inf values.
    # https://discuss.pytorch.org/t/gram-matrix-in-mixed-precision/166800/2
    # x = torch.clamp(x, max=1.0e2, min=-1.0e2)
    # x[x > 1.0e2] = 1.0e2
    # x[x < -1.0e2] = -1.0e2

    G = torch.mm(x, x.T)
    G = torch.clamp(G, -64990.0, 64990.0)
    # normalize by total elements
    result = G.div(b * c * w * h)
    return result



def divisible(dim):
    '''
    Make width and height divisible by 32
    '''
    width, height = dim
    return width - (width % 32), height - (height % 32)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width and height:
        return cv2.resize(image, divisible((width, height)),  interpolation=inter)

    if width is None and height is None:
        return cv2.resize(image, divisible((w, h)),  interpolation=inter)

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, divisible(dim), interpolation=inter)


def normalize_input(images):
    '''
    [0, 255] -> [-1, 1]
    '''
    return images / 127.5 - 1.0


def denormalize_input(images, dtype=None):
    '''
    [-1, 1] -> [0, 255]
    '''
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = images.type(dtype)
        else:
            # numpy.ndarray
            images = images.astype(dtype)

    return images


def preprocess_images(images):
    '''
    Preprocess image for inference

    @Arguments:
        - images: np.ndarray

    @Returns
        - images: torch.tensor
    '''
    images = images.astype(np.float32)

    # Normalize to [-1, 1]
    images = normalize_input(images)
    images = torch.from_numpy(images)

    # Add batch dim
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # channel first
    images = images.permute(0, 3, 1, 2)

    return images

def compute_data_mean(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    print(f"Compute mean (R, G, B) from {len(image_files)} images")

    for img_file in tqdm(image_files):
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[...,::-1]  # Convert to BGR for training


if __name__ == '__main__':
    t = torch.rand(2, 14, 32, 32)

    with torch.autocast("cpu"):
        print(gram(t))
