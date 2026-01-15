"""Cross-entropy and Dice losses."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Compute Dice loss for multi-class segmentation.

    Args:
        logits: [B, C, H, W] logits.
        targets: [B, H, W] integer labels.
        num_classes: Number of classes.
        ignore_index: Ignore label.

    Returns:
        Dice loss.
    """
    probs = torch.softmax(logits, dim=1)
    targets_safe = targets.clone()
    invalid = (targets_safe == ignore_index) | (targets_safe < 0) | (targets_safe >= num_classes)
    targets_safe = targets_safe.clamp(min=0, max=num_classes - 1)
    targets_onehot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = (~invalid).unsqueeze(1).float()
    probs = probs * valid
    targets_onehot = targets_onehot * valid
    dims = (0, 2, 3)
    inter = (probs * targets_onehot).sum(dims)
    denom = probs.sum(dims) + targets_onehot.sum(dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    """Combined cross-entropy and Dice loss."""

    def __init__(self, num_classes: int, ignore_index: int = 255, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_safe = targets.clone()
        targets_safe[targets_safe < 0] = self.ignore_index
        targets_safe[targets_safe >= self.num_classes] = self.ignore_index
        ce = F.cross_entropy(logits, targets_safe, ignore_index=self.ignore_index)
        d = dice_loss(logits, targets_safe, self.num_classes, self.ignore_index)
        return self.ce_weight * ce + self.dice_weight * d
