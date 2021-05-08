import torch
import matplotlib.pyplot as plt
import gc
import os

_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).cuda().float()


def gram(input):
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image, channel_last=False):
    '''
    https://en.wikipedia.org/wiki/YUV
    '''

    # Conver to channel first
    if channel_last:
        image = image.permute(2, 0, 1).type(torch.float32)

    yuv_img = torch.tensordot(image.float(), _rgb_to_yuv_kernel, dims=([0], [0]))

    if not channel_last:
        yuv_img = yuv_img.permute(0, 1, 2)

    return yuv_img


def rgb_to_yuv_batch(images, channel_last=False):
    '''
    Extend of rgb_to_yuv to run on batch
    '''

    if channel_last:
        images = images.permute(0, 3, 1, 2).type(torch.float32)

    yuv_img = torch.tensordot(images.float(), _rgb_to_yuv_kernel, dims=([1], [0]))

    if not channel_last:
        yuv_img = yuv_img.permute(0, 3, 1 ,2)

    return yuv_img


def show_images(images, rows=2):
    columns = len(images)
    height = 4
    width = height * columns // 2

    fig = plt.figure(figsize=(height, width))

    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.show()


def save_checkpoint(model, optimizer, epoch, args):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}.pth')
    torch.save(checkpoint, path)


def load_checkpoint(model, args):
    path = os.path.join(args.checkpoint_dir, f'{model.name}.pth')
    checkpoint = torch.load(path,  map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return epoch


class DictToObject(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)
