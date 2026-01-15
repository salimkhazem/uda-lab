"""SSDD SAR ship segmentation dataset."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base_dataset import BaseSegDataset


class SSDDDataset(BaseSegDataset):
    """SSDD SAR ship segmentation dataset."""

    @staticmethod
    def get_splits(root: str) -> Dict[str, List[Tuple[str, str]]]:
        splits_txt = os.path.join(root, "splits")
        images_dir = os.path.join(root, "images")
        masks_dir = os.path.join(root, "masks")
        if os.path.isdir(splits_txt) and os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            splits = {}
            for split in ["train", "val", "test"]:
                txt_path = os.path.join(splits_txt, f"{split}.txt")
                if not os.path.exists(txt_path):
                    continue
                pairs = []
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        name = line.strip()
                        if not name:
                            continue
                        img_path = os.path.join(images_dir, name)
                        mask_path = os.path.join(masks_dir, name)
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            pairs.append((img_path, mask_path))
                splits[split] = pairs
            if splits:
                return splits

        expected = [
            ("train", "images/train", "masks/train"),
            ("val", "images/val", "masks/val"),
            ("test", "images/test", "masks/test"),
        ]
        splits = {}
        for split, img_dir, mask_dir in expected:
            img_path = os.path.join(root, img_dir)
            mask_path = os.path.join(root, mask_dir)
            if not os.path.isdir(img_path) or not os.path.isdir(mask_path):
                raise FileNotFoundError(
                    "SSDD data not found. Expected structure: "
                    "data/ssdd/images/{train,val,test} and data/ssdd/masks/{train,val,test} "
                    "or data/ssdd/splits/*.txt with images/ and masks/."
                )
            splits[split] = BaseSegDataset.list_pairs(img_path, mask_path)
        return splits
