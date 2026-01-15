"""Exponential moving average for model weights."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    """EMA wrapper for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights."""
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
