from imgaug.parameters import Positive
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.
    :return:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss,
                              weight=weight,
                              reduction=reduction,
                              avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels),
                         as_tuple=False).squeeze()

    if inds.numel() > 0:
        bin_label_weights = None

    else:
        bin_label_weights = label_weights.view(-1, 0).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.

    :return:
        torch.Tensor: The calculated loss

    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(pred,
                                              label.float(),
                                              pos_weight=class_weight,
                                              reduction="none")

    loss = weight_reduce_loss(loss,
                              weight,
                              reduction=reduction,
                              avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the  CrossEntropy loss for masks.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.

    :return:
        torch.Tensor: The calculated loss
    """
    assert reduction == 'mean' and avg_factor is None

    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice,
                                              target,
                                              weight=class_weight,
                                              reduction='mean')[None]


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """Cross EntropyLoss


        :param use_sigmoid: (bool,optional) whether the prediction uses sigmoid
        :param use_mask:  (bool,optional) whether the prediction uses mask
        :param reduction:  (str,optional) Defaults to 'mean'.
        :param class_weight: (list[float],optional) Weight of each class.
        :param loss_weight:  (float,optional) weight of the loss.Defaults to 1.0

        """

        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy

        if self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        :param cls_score: (torch.Tensor)
        :param label:
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :param kwargs:
        :return:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss_cls


class OHEMLoss(nn.Module):
    def __init__(self, ohem=True, ohem_ratio=3.0):
        super(OHEMLoss, self).__init__()
        self.ohem_ratio = ohem_ratio
        self.eps = 1e-6

    def forward(self, cls_score, label, mask):
        pos = (label * mask).byte()
        neg = ((1 - label) * mask).byte()

        pos_cnt = pos.float().sum()
        neg_cnt = min(neg.float().sum(), pos_cnt * self.ohem_ratio)

        pos_cnt, neg_cnt = int(pos_cnt), int(neg_cnt)
        
        loss =nn.functional.binary_cross_entropy(cls_score, label,reduction='none')

        pos_loss = loss * pos.float()
        neg_loss = loss * neg.float()

        neg_loss, _ = neg_loss.view(-1).topk(neg_cnt)

        ohem_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_cnt + neg_cnt +
                                                         self.eps)

        return ohem_loss
    


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        # negative_loss, _ = torch.topk(negative_loss.view(-1).contiguous(), negative_count)
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss
