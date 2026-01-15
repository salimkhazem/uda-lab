"""DeepLabv3+ wrapper."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ segmentation model wrapper."""

    def __init__(self, num_classes: int, pretrained: bool = True, backbone: str = "resnet50") -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Only resnet50 backbone is supported in this wrapper.")
        self.model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features for alignment."""
        features = self.model.backbone(x)
        if isinstance(features, dict):
            return list(features.values())[-1]
        return features
