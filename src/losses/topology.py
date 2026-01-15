"""Topology-preserving losses using soft skeletonization."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(prob_map: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Compute soft skeleton of a probability map.

    Args:
        prob_map: Probability map [B, 1, H, W].
        iters: Iterations for thinning.

    Returns:
        Soft skeleton map.
    """
    skel = torch.zeros_like(prob_map)
    img = prob_map
    for _ in range(iters):
        opened = _soft_open(img)
        delta = torch.relu(img - opened)
        skel = skel + torch.relu(delta - skel)
        img = _soft_erode(img)
    return skel


def cldice_loss(pred_prob: torch.Tensor, gt_onehot: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Compute clDice loss for binary or multi-class masks.

    Args:
        pred_prob: Predicted probabilities [B, C, H, W].
        gt_onehot: Ground-truth onehot [B, C, H, W].
        iters: Skeletonization iterations.

    Returns:
        clDice loss.
    """
    eps = 1e-6
    if pred_prob.shape[1] == 1:
        pred_prob = torch.sigmoid(pred_prob)
        skel_pred = soft_skeletonize(pred_prob, iters)
        skel_gt = soft_skeletonize(gt_onehot.float(), iters)
        tprec = (skel_pred * gt_onehot).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + eps)
        tsens = (skel_gt * pred_prob).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + eps)
        cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
        return 1 - cldice.mean()

    # Multi-class: ignore background class (0)
    cls_losses = []
    for c in range(1, pred_prob.shape[1]):
        p = pred_prob[:, c : c + 1]
        g = gt_onehot[:, c : c + 1].float()
        skel_pred = soft_skeletonize(p, iters)
        skel_gt = soft_skeletonize(g, iters)
        tprec = (skel_pred * g).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + eps)
        tsens = (skel_gt * p).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + eps)
        cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
        cls_losses.append(1 - cldice.mean())
    if not cls_losses:
        return torch.tensor(0.0, device=pred_prob.device)
    return torch.stack(cls_losses).mean()


def topology_consistency_loss(pred_prob: torch.Tensor, pseudo_prob: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Topology consistency between prediction and pseudo-probabilities.

    Args:
        pred_prob: Predicted probabilities [B, C, H, W].
        pseudo_prob: Pseudo-label probabilities [B, C, H, W].
        iters: Skeletonization iterations.
    """
    return cldice_loss(pred_prob, pseudo_prob, iters=iters)


class TopologyLoss(nn.Module):
    """Topology loss wrapper."""

    def __init__(self, iters: int = 10) -> None:
        super().__init__()
        self.iters = iters

    def forward(self, pred_prob: torch.Tensor, gt_onehot: torch.Tensor) -> torch.Tensor:
        return cldice_loss(pred_prob, gt_onehot, iters=self.iters)
