import torch
import matplotlib.pyplot as plt
import gc
import os
from time import gmtime, strftime


_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()

if torch.cuda.is_available():
    _rgb_to_yuv_kernel = _rgb_to_yuv_kernel.cuda()
    vgg_std = vgg_std.cuda()
    vgg_mean = vgg_mean.cuda()


def gram(input):
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image, channel_last=False):
    '''
    https://en.wikipedia.org/wiki/YUV

    output: Image of shape (H, W, C) (channel last)
    '''
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    # Conver to channel first
    if channel_last:
        image = image.permute(2, 0, 1).type(torch.float32)

    yuv_img = torch.tensordot(image.float(), _rgb_to_yuv_kernel, dims=([0], [0]))

    return yuv_img


def rgb_to_yuv_batch(images, channel_last=False):
    '''
    Extend of rgb_to_yuv to run on batch

    output: Image of shape (B, H, W, C) (channel last)
    '''
    images = (images + 1.0)  / 2.0
    if channel_last:
        images = images.permute(0, 3, 1, 2).type(torch.float32)

    yuv_img = torch.tensordot(images.float(), _rgb_to_yuv_kernel, dims=([1], [0]))

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
        plt.savefig(f'/content/{now}.png')
    else:
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


class DefaultArgs():
    dataset ='Hayao'
    data_dir ='/content'
    epochs =10
    batch_size =16
    checkpoint_dir ='/content/checkpoints'
    save_image_dir ='/content/images'
    display_image =True
    save_interval =2
    debug_samples =0
    lr_g =0.001
    lr_d =0.002
    wadv =300.0
    wcon =1.5
    wgra =3
    wcol =10