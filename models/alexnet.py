import torch.nn as nn

from core.dataloader import set_mean_std
from .blocks import ConvBlock, LinearBlock, NormalizeLayer


class AlexNet(nn.Module):
    def __init__(self, dataset, act, bn) -> None:
        super().__init__()
        mean, std = set_mean_std(dataset)
        self.norm_layer = NormalizeLayer(mean, std)
        self.layers = nn.Sequential(
            ConvBlock(3, 64, act=act, bn=bn, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(64, 192, act=act, bn=bn, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(192, 384, act=act, bn=bn, kernel_size=3, padding=1),
            ConvBlock(384, 256, act=act, bn=bn, kernel_size=3, padding=1),
            ConvBlock(256, 256, act=act, bn=bn, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LinearBlock(256 * 1 * 1, 4096),
            LinearBlock(4096, 4096, bn=bn, act=act),
            LinearBlock(4096, self.set_output_num(dataset), bn=bn, act=None),
        )

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.layers(x)
        return x

    @staticmethod
    def set_output_num(dataset):
        if dataset.lower() == "cifar10":
            num_cls = 10
        elif dataset.lower() == "cifar100":
            num_cls = 100
        elif dataset.lower() == "imagenet":
            num_cls = 1000
        else:
            raise NameError()
        return num_cls
