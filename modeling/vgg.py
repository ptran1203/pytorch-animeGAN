import torchvision.models as models
import torch.nn as nn

def get_vgg19(last_layer='conv4_4'):
    vgg = models.vgg19(pretrained=True).features.eval()
    model_list = []

    i = 0
    j = 1
    for layer in vgg.children():
        if isinstance(layer, nn.MaxPool2d):
            i = 0
            j += 1

        elif isinstance(layer, nn.Conv2d):
            i += 1

        name = f'conv{j}_{i}'

        if name == last_layer:
            model_list.append(layer)
            break

        model_list.append(layer)


    model = nn.Sequential(*model_list)
    return model
