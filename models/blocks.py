import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import init

from core.utils import set_activation


class BaseBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weight(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias.data)
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias.data)


class LinearBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, act="relu", bn=True):
        super().__init__()

        self.LT = nn.Linear(in_channels, out_channels)
        self.BN = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.Act = set_activation(act)
        self.init_weight()

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class ConvBlock(BaseBlock):
    def __init__(
            self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, act="relu", bn=True, **kwargs
    ):
        super().__init__()
        self.LT = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.BN = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.Act = set_activation(act)
        self.init_weight()

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, x: torch.tensor):
        device = x.device
        (batch_size, num_channels, height, width) = x.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(device)
        return (x - means) / sds


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels, planes, stride=1, act="relu", bn=True, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.act1 = set_activation(act)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        self.act2 = set_activation(act)
        if in_channels != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes) if bn else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act1(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            planes: int,
            stride=1,
            act="relu",
            bn=True,
            bias=False,
    ) -> None:
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.act1 = set_activation(act)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.act2 = set_activation(act)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if bn else nn.Identity()

        if in_channels != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, planes * self.expansion, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes * self.expansion) if bn else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

        self.act = set_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
