import requests
import urllib.request
import cv2

HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

def read_image(path):
    """
    Read image from given path
    """

    if any(path.startswith(p) for p in HTTP_PREFIXES):
        urllib.request.urlretrieve(path, "temp.jpg")
        path = "temp.jpg"

    return cv2.imread(path)[: ,: ,::-1]

class DefaultArgs:
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
