import torch.nn as nn

from core.dataloader import set_mean_std
from .blocks import ConvBlock, LinearBlock, NormalizeLayer


class VGG(nn.Module):
    def __init__(self, dataset, act, bn, layers):
        super().__init__()
        mean, std = set_mean_std(dataset)
        self.norm_layer = NormalizeLayer(mean, std)
        self.layers = nn.Sequential(*self._build_features(layers, act, bn), *self._build_classifier(dataset, act, bn))

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.layers(x)
        return x

    def _build_features(self, layers, act, bn):
        return [
            *self._build_block(3, 64, layers[0], act, bn),
            *self._build_block(64, 128, layers[1], act, bn),
            *self._build_block(128, 256, layers[2], act, bn),
            *self._build_block(256, 512, layers[3], act, bn),
            *self._build_block(512, 512, layers[4], act, bn),
        ]

    @staticmethod
    def _build_classifier(dataset, act, bn):
        if dataset.lower() == "cifar10":
            output_size, num_cls = 1, 10
        elif dataset.lower() == "cifar100":
            output_size, num_cls = 1, 100
        elif dataset.lower() == "imagenet":
            output_size, num_cls = 7, 1000
        else:
            raise NameError()
        return [
            nn.AdaptiveAvgPool2d((output_size, output_size)),
            nn.Flatten(),
            LinearBlock(512 * output_size * output_size, 4096, bn=bn, act=act),
            nn.Dropout(p=0.5),
            LinearBlock(4096, 4096, bn=bn, act=act),
            nn.Dropout(p=0.5),
            LinearBlock(4096, num_cls, bn=bn, act=None),
        ]

    @staticmethod
    def _build_block(in_channel, out_channel, num_layers, act, bn):
        layers = [ConvBlock(in_channel, out_channel, act=act, bn=bn)]
        layers += [ConvBlock(out_channel, out_channel, act=act, bn=bn) for _ in range(num_layers - 1)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return layers


def make_vgg(model, dataset, act, bn):
    if model.lower() == 'vgg11':
        net = VGG(dataset, act, bn, [1, 1, 2, 2, 2])
    elif model.lower() == "vgg13":
        net = VGG(dataset, act, bn, [2, 2, 2, 2, 2])
    elif model.lower() == "vgg16":
        net = VGG(dataset, act, bn, [2, 2, 3, 3, 3])
    elif model.lower() == "vgg19":
        net = VGG(dataset, act, bn, [2, 2, 4, 4, 4])
    else:
        raise NameError(f"No model named {model}")
    return net
