"""Consistency losses."""

from __future__ import annotations

import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """L1 consistency loss between logits or probabilities."""

    def __init__(self, use_probs: bool = True) -> None:
        super().__init__()
        self.use_probs = use_probs
        self.l1 = nn.L1Loss()

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        if self.use_probs:
            probs_a = torch.softmax(logits_a, dim=1)
            probs_b = torch.softmax(logits_b, dim=1)
            return self.l1(probs_a, probs_b)
        return self.l1(logits_a, logits_b)
