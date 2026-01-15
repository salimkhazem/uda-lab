"""Domain adversarial loss with gradient reversal."""

from __future__ import annotations

import torch
from torch import nn


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRL.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """Simple domain classifier on feature maps."""

    def __init__(self, in_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss using GRL and BCE."""

    def __init__(self, in_channels: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.grl = GradientReversal(alpha=alpha)
        self.clf = DomainClassifier(in_channels)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        feats = self.grl(features)
        logits = self.clf(feats)
        return self.bce(logits, domain_labels.float())
