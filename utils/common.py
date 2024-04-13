import torch
import gc
import os
import torch.nn as nn
import urllib.request
import cv2
from tqdm import tqdm

HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

SUPPORT_WEIGHTS = {
    'hayao',
    'shinkai',
}

ASSET_HOST = 'https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0'


def read_image(path):
    """
    Read image from given path
    """

    if any(path.startswith(p) for p in HTTP_PREFIXES):
        urllib.request.urlretrieve(path, "temp.jpg")
        path = "temp.jpg"

    return cv2.imread(path)[: ,: ,::-1]


def save_checkpoint(model, path, optimizer=None, epoch=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is  not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(model, path, optimizer=None) -> int:
    state_dict = load_state_dict(path)
    model.load_state_dict(state_dict['model_state_dict'], strict=True)
    if optimizer is not None and 'optimizer_state_dict' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    epoch = state_dict.get('epoch', 0)
    return epoch


def load_state_dict(weight) -> dict:
    if weight.lower() in SUPPORT_WEIGHTS:
        weight = _download_weight(weight)

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(weight, map_location=map_location)

    return state_dict


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


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DownloadProgressBar(tqdm):
    '''
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    '''
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_weight(weight):
    '''
    Download weight and save to local file
    '''
    filename = f'generator_{weight.lower()}.pth'
    os.makedirs('.cache', exist_ok=True)
    url = f'{ASSET_HOST}/{filename}'
    save_path = f'.cache/{filename}'

    if os.path.isfile(save_path):
        return save_path

    desc = f'Downloading {url} to {save_path}'
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    return save_path

