"""Albumentations transforms."""

from __future__ import annotations

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(img_size: Tuple[int, int], is_train: bool = True):
    """Build training or evaluation transforms.

    Args:
        img_size: (height, width) target size.
        is_train: Whether to build training transforms.
    """
    h, w = img_size
    if is_train:
        return A.Compose(
            [
                A.Resize(height=h, width=w),
                A.PadIfNeeded(min_height=h, min_width=w, border_mode=0),
                A.RandomCrop(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Affine(
                    translate_percent=0.1,
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    p=0.3,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def build_weak_strong(img_size: Tuple[int, int]):
    """Build weak and strong augmentation pipelines."""
    weak = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=0),
            A.RandomCrop(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    strong = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=0),
            A.RandomCrop(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.8),
            A.GaussianBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return weak, strong
