import torch
import matplotlib.pyplot as plt
import gc
import os
import cv2
import torch.nn as nn
from time import gmtime, strftime


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


def show_images(images, rows=2, height=10, save=True):
    cols = len(images)
    width = height * cols // 2

    fig = plt.figure(figsize=(height, width))

    for i in range(1, rows * cols + 1):
        if i > cols:
            break

        img = images[i - 1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)

    if save:
        now = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        plt.savefig(f'/content/generated/{now}.png')
    else:
        plt.show()


def save_checkpoint(model, optimizer, epoch, args, posfix=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}{posfix}.pth')
    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_dir, posfix=''):
    path = os.path.join(checkpoint_dir, f'{model.name}{posfix}.pth')
    checkpoint = torch.load(path,  map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return epoch


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


def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass


class DefaultArgs():
    dataset ='Hayao'
    data_dir ='/content'
    epochs = 10
    batch_size = 1
    checkpoint_dir ='/content/checkpoints'
    save_image_dir ='/content/images'
    display_image =True
    save_interval =2
    debug_samples =0
    lr_g = 0.001
    lr_d = 0.002
    wadv = 300.0
    wcon = 1.5
    wgra = 3
    wcol = 10
    use_sn = False
