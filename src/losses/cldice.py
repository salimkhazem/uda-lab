"""clDice loss wrapper."""

from __future__ import annotations

import torch
from torch import nn

from .topology import cldice_loss


class CLDiceLoss(nn.Module):
    """clDice loss for topology-aware supervision."""

    def __init__(self, iters: int = 10) -> None:
        super().__init__()
        self.iters = iters

    def forward(self, pred_prob: torch.Tensor, gt_onehot: torch.Tensor) -> torch.Tensor:
        return cldice_loss(pred_prob, gt_onehot, iters=self.iters)
