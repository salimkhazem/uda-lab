"""Model factory."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .deeplabv3plus import DeepLabV3Plus
from .segformer import SegFormer
from .swin_unet import SwinUNet
from .unet import UNet


class HRDALike(nn.Module):
    """Lightweight HRDA-like multi-resolution fusion."""

    def __init__(self, num_classes: int, backbone: str = "resnet50", high_res_scale: float = 1.5) -> None:
        super().__init__()
        self.high_res_scale = high_res_scale
        self.low_res = DeepLabV3Plus(num_classes=num_classes, pretrained=True, backbone=backbone)
        self.high_res = UNet(in_channels=3, num_classes=num_classes, base_channels=32)
        self.fuse = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_res(x)
        if self.high_res_scale != 1.0:
            high_in = torch.nn.functional.interpolate(
                x, scale_factor=self.high_res_scale, mode="bilinear", align_corners=False
            )
        else:
            high_in = x
        high = self.high_res(high_in)
        if high.shape[-2:] != low.shape[-2:]:
            high = torch.nn.functional.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([low, high], dim=1)
        return self.fuse(fused)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.low_res.get_features(x)


def create_model(cfg: Dict) -> nn.Module:
    """Create model from config."""
    name = cfg["model"]["name"]
    num_classes = cfg["dataset"]["num_classes"]
    if name == "unet":
        return UNet(cfg["model"].get("in_channels", 3), num_classes, cfg["model"].get("base_channels", 64))
    if name == "deeplabv3p":
        return DeepLabV3Plus(
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
            backbone=cfg["model"].get("backbone", "resnet50"),
        )
    if name == "segformer":
        return SegFormer(
            num_classes=num_classes,
            backbone=cfg["model"].get("backbone", "mit_b0"),
            pretrained=cfg["model"].get("pretrained", True),
        )
    if name == "swin_unet":
        return SwinUNet(
            num_classes=num_classes,
            backbone=cfg["model"].get("backbone", "swin_tiny_patch4_window7_224"),
            pretrained=cfg["model"].get("pretrained", True),
        )
    if name == "hrda_like":
        return HRDALike(
            num_classes=num_classes,
            backbone=cfg["model"].get("backbone", "resnet50"),
            high_res_scale=cfg["model"].get("high_res_scale", 1.5),
        )
    raise ValueError(f"Unknown model: {name}")
