from typing import List, Type, Union

import torch.nn as nn
from torch import Tensor

from core.dataloader import set_mean_std
from models.blocks import BasicBlock, Bottleneck, ConvBlock, NormalizeLayer, LinearBlock


class ResNet(nn.Module):
    def __init__(
            self,
            dataset: str,
            act: str,
            bn: bool,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            groups: int = 1,
            width_per_group: int = 64,
    ) -> None:
        super().__init__()
        self.act = act
        self.bn = bn

        mean, std = set_mean_std(dataset)
        self.norm_layer = NormalizeLayer(mean, std)

        self.in_channels = 64

        self.groups = groups
        self.base_width = width_per_group

        if dataset.lower() == "imagenet":
            self.conv1 = ConvBlock(3, self.in_channels, kernel_size=7, stride=2, padding=3, bn=bn, act=act)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = ConvBlock(3, self.in_channels, kernel_size=3, stride=2, padding=1, bn=bn, act=act)
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, layers[0], 64, act, bn)
        self.layer2 = self._make_layer(block, layers[1], 128, act, bn, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, act, bn, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, act, bn, stride=2)
        self.fc = self._make_classifier(dataset, block)

        self.layers = [
            self.conv1,
            self.maxpool,
            *list(self.layer1),
            *list(self.layer2),
            *list(self.layer3),
            *list(self.layer4),
            *list(self.fc)
        ]
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            num_blocks: int,
            planes: int,
            act: str,
            bn: bool,
            stride: int = 1,
    ) -> nn.Sequential:
        layers = []

        layers.append(block(in_channels=self.in_channels, planes=planes, stride=stride, act=act, bn=bn))
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels=self.in_channels, planes=planes, act=act, bn=bn))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.norm_layer(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _make_classifier(self, dataset, block):
        if dataset.lower() == "cifar10":
            output_size, num_cls = 1, 10
        elif dataset.lower() == "cifar100":
            output_size, num_cls = 1, 100
        elif dataset.lower() == "imagenet":
            output_size, num_cls = 7, 1000
        else:
            raise NameError()
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        fc = LinearBlock(512 * block.expansion, num_cls, bn=self.bn, act=None)
        return nn.Sequential(avgpool, flatten, fc)


def make_resnet(model, dataset, act, bn):
    if model.lower() == "resnet18":
        net = ResNet(dataset, act, bn, BasicBlock, [2, 2, 2, 2])
    elif model.lower() == "resnet34":
        net = ResNet(dataset, act, bn, BasicBlock, [3, 4, 6, 3])
    elif model.lower() == "resnet50":
        net = ResNet(dataset, act, bn, Bottleneck, [3, 4, 6, 3])
    elif model.lower() == "resnet101":
        net = ResNet(dataset, act, bn, Bottleneck, [3, 4, 23, 3])
    else:
        raise NameError(f"No model named {model}")
    return net
