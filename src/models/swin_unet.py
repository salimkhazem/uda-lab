"""Swin-UNet style model using timm features."""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import timm


class UpBlock(nn.Module):
    """Upsample + conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    """Swin backbone with UNet-style decoder."""

    def __init__(self, num_classes: int, backbone: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        self._enc_img_size = getattr(self.encoder, "img_size", None)
        if self._enc_img_size is None and hasattr(self.encoder, "patch_embed"):
            self._enc_img_size = getattr(self.encoder.patch_embed, "img_size", None)
        channels = self.encoder.feature_info.channels()
        self.up3 = UpBlock(channels[3], channels[2], channels[2])
        self.up2 = UpBlock(channels[2], channels[1], channels[1])
        self.up1 = UpBlock(channels[1], channels[0], channels[0])
        self.up0 = nn.ConvTranspose2d(channels[0], channels[0] // 2, kernel_size=2, stride=2)
        self.head = nn.Conv2d(channels[0] // 2, num_classes, kernel_size=1)

    def _to_bchw(self, feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        """Convert (B,H,W,C) -> (B,C,H,W). Swin always outputs BHWC."""
        if feat.dim() != 4:
            return feat
        # If last dim matches expected channels, it's BHWC -> permute
        if feat.shape[-1] == expected_c:
            return feat.permute(0, 3, 1, 2).contiguous()
        # Otherwise assume already BCHW
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        if self._enc_img_size is not None:
            target_h, target_w = int(self._enc_img_size[0]), int(self._enc_img_size[1])
            if orig_size != (target_h, target_w):
                x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        feats = self.encoder(x)
        # Swin outputs (B,H,W,C), convert to (B,C,H,W)
        channels = self.encoder.feature_info.channels()
        feats = [self._to_bchw(f, c) for f, c in zip(feats, channels)]
        x3 = self.up3(feats[3], feats[2])
        x2 = self.up2(x3, feats[1])
        x1 = self.up1(x2, feats[0])
        x0 = self.up0(x1)
        out = self.head(x0)
        if out.shape[-2:] != orig_size:
            out = torch.nn.functional.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return deepest feature map in (B,C,H,W) format."""
        if self._enc_img_size is not None:
            target_h, target_w = int(self._enc_img_size[0]), int(self._enc_img_size[1])
            if x.shape[-2:] != (target_h, target_w):
                x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        feats = self.encoder(x)
        channels = self.encoder.feature_info.channels()
        return self._to_bchw(feats[-1], channels[-1])
