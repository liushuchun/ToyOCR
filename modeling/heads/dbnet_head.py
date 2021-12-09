from typing_extensions import OrderedDict
import torch
from torch import nn


class DBHead(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 k=10,
                 smooth=False,
                 bias=False):
        super(DBHead, self).__init__()

        self.binarization = nn.Sequential(
            nn.Conv2d(inner_channels,
                      inner_channels // 4,
                      3,
                      padding=1,
                      bias=bias), nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels,
                      inner_channels // 4,
                      3,
                      padding=1,
                      bias=bias), nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4,
                                inner_channels // 4,
                                smooth=smooth,
                                bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4,
                                1,
                                smooth=smooth,
                                bias=bias), nn.Sigmoid())

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, feature):
        binary = self.binarization(feature)

        thresh = self.thresh(feature)
        thresh_binary = self.step_function(binary, thresh)

        result = OrderedDict(binary=binary,
                             thresh=thresh,
                             thresh_binary=thresh_binary)

        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
