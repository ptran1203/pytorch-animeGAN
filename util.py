import torch


_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
])



def gram(input):
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image, channel_last=False):
    '''
    https://en.wikipedia.org/wiki/YUV
    https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/ops/image_ops_impl.py#L3759-L3782
    '''

    # Conver to channel first
    if channel_last:
        image = image.permute(2, 0, 1).type(torch.float32)
        print(image.shape)

    return torch.tensordot(image, _rgb_to_yuv_kernel, dims=([0], [0]))
