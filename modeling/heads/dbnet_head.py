from typing_extensions import OrderedDict
import torch
from torch import nn
from ..losses import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBNetHead(nn.Module):
    def __init__(self,
                 cfg,
                 in_channels,
                 inner_channels=256,
                 k=10,
                 smooth=False,
                 bias=False,
                 alpha=1.0,
                 beta=10,
                 ohem_ratio=3,
                 reduction='mean',
                 eps=1e-6):
        super(DBNetHead, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

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

    def losses(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'],
                                         batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps,
                                           batch['threshold_map'],
                                           batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps,
                       loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'],
                                              batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
