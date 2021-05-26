from .common import *
from .image_processing import *

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
    wadvg = 300.0
    wadvd = 300.0
    wcon = 1.5
    wgra = 3
    wcol = 10
    use_sn = False
