"""DeepGlobe Roads dataset."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base_dataset import BaseSegDataset


class DeepGlobeRoadsDataset(BaseSegDataset):
    """DeepGlobe road extraction dataset."""

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
                    "DeepGlobe data not found. Expected structure: "
                    "data/deepglobe/images/{train,val,test} and data/deepglobe/masks/{train,val,test}."
                )
            splits[split] = BaseSegDataset.list_pairs(img_path, mask_path)
        return splits
