"""UNet model."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolution block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Lightweight UNet for segmentation."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64) -> None:
        super().__init__()
        chs = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down1 = ConvBlock(in_channels, chs[0])
        self.down2 = ConvBlock(chs[0], chs[1])
        self.down3 = ConvBlock(chs[1], chs[2])
        self.down4 = ConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(chs[3], chs[3] * 2)

        self.up3 = nn.ConvTranspose2d(chs[3] * 2, chs[3], kernel_size=2, stride=2)
        self.conv3 = ConvBlock(chs[3] * 2, chs[3])
        self.up2 = nn.ConvTranspose2d(chs[3], chs[2], kernel_size=2, stride=2)
        self.conv2 = ConvBlock(chs[2] * 2, chs[2])
        self.up1 = nn.ConvTranspose2d(chs[2], chs[1], kernel_size=2, stride=2)
        self.conv1 = ConvBlock(chs[1] * 2, chs[1])
        self.up0 = nn.ConvTranspose2d(chs[1], chs[0], kernel_size=2, stride=2)
        self.conv0 = ConvBlock(chs[0] * 2, chs[0])

        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        u3 = self.up3(b)
        u3 = self.conv3(torch.cat([u3, d4], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d3], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d2], dim=1))
        u0 = self.up0(u1)
        u0 = self.conv0(torch.cat([u0, d1], dim=1))
        return self.head(u0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate feature map for domain alignment."""
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        return b
