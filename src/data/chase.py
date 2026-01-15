"""CHASEDB1 dataset."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base_dataset import BaseSegDataset


class CHASEDataset(BaseSegDataset):
    """CHASEDB1 retinal vessel segmentation dataset."""

    @staticmethod
    def get_splits(root: str) -> Dict[str, List[Tuple[str, str]]]:
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
                    "CHASE data not found. Expected structure: "
                    "data/chase/images/{train,val,test} and data/chase/masks/{train,val,test}."
                )
            splits[split] = BaseSegDataset.list_pairs(img_path, mask_path)
        return splits
