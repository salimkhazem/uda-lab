"""Base dataset classes for segmentation."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from ..utils.image import read_image, read_mask


class BaseSegmentationDataset(Dataset):
    """Base dataset with common utilities for segmentation."""

    def __init__(
        self,
        root: str,
        img_size: Tuple[int, int],
        transform=None,
        ignore_index: int = 255,
        num_classes: int = 2,
        label_map: np.ndarray | None = None,
    ) -> None:
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.label_map = label_map

    def _apply_label_map(self, mask: np.ndarray) -> np.ndarray:
        if self.label_map is None:
            return mask
        return self.label_map[mask]

    def _apply_transform(self, image: np.ndarray, mask: np.ndarray):
        if self.transform is None:
            return image, mask
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 0 and isinstance(image, np.ndarray) and image.ndim >= 2:
            mask = np.full((image.shape[0], image.shape[1]), mask, dtype=np.int32)
        else:
            mask = mask.astype(np.int32, copy=False)
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def _normalize_mask(self, mask):
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3:
                return mask[..., 0]
            return mask
        try:
            import torch

            if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                if mask.shape[0] == 1:
                    return mask[0].long()
                if mask.shape[-1] == 1:
                    return mask.squeeze(-1).long()
                if mask.shape[-1] <= 4 and mask.shape[0] > 4:
                    return mask[..., 0].long()
                if mask.shape[0] <= 4 and mask.shape[-1] > 4:
                    return mask[0].long()
                return mask[..., 0].long()
        except Exception:
            pass
        return mask


class BaseSegDataset(BaseSegmentationDataset):
    """Simple dataset with a list of (image, mask) pairs."""

    def __init__(
        self,
        root: str,
        samples: List[Tuple[str, str]],
        img_size: Tuple[int, int],
        transform=None,
        ignore_index: int = 255,
        num_classes: int = 2,
        label_map: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            root=root,
            img_size=img_size,
            transform=transform,
            ignore_index=ignore_index,
            num_classes=num_classes,
            label_map=label_map,
        )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img_path, mask_path = self.samples[idx]
        image = read_image(img_path)
        mask = read_mask(mask_path)
        mask = self._apply_label_map(mask)
        image, mask = self._apply_transform(image, mask)
        mask = self._normalize_mask(mask)
        return {
            "image": image,
            "mask": mask.long() if hasattr(mask, "long") else mask,
            "meta": {"image_path": img_path, "mask_path": mask_path},
        }

    @staticmethod
    def list_pairs(
        image_dir: str, mask_dir: str, exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "tif", "tiff")
    ) -> List[Tuple[str, str]]:
        images = []
        for root, _, files in os.walk(image_dir):
            for name in files:
                if name.lower().endswith(exts):
                    images.append(os.path.join(root, name))
        images = sorted(images)
        pairs: List[Tuple[str, str]] = []
        for img_path in images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None
            for ext in exts:
                candidate = os.path.join(mask_dir, base + "." + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                continue
            pairs.append((img_path, mask_path))
        return pairs

    @staticmethod
    def get_splits(root: str) -> Dict[str, List[Tuple[str, str]]]:
        raise NotImplementedError
