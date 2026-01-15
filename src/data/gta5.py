"""GTA5 dataset."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base_dataset import BaseSegDataset


class GTA5Dataset(BaseSegDataset):
    """GTA5 semantic segmentation dataset."""

    @staticmethod
    def get_splits(root: str) -> Dict[str, List[Tuple[str, str]]]:
        expected = [
            ("train", "images/train", "labels/train"),
            ("val", "images/val", "labels/val"),
        ]
        splits = {}
        ok = True
        for split, img_dir, mask_dir in expected:
            img_path = os.path.join(root, img_dir)
            mask_path = os.path.join(root, mask_dir)
            if not os.path.isdir(img_path) or not os.path.isdir(mask_path):
                ok = False
                break
            splits[split] = BaseSegDataset.list_pairs(img_path, mask_path)
        if ok:
            return splits

        # Fallback to uppercase GTA5 folder
        alt_root = os.path.join(os.path.dirname(root), "GTA5")
        if os.path.isdir(alt_root):
            splits = {}
            for split, img_dir, mask_dir in expected:
                img_path = os.path.join(alt_root, img_dir)
                mask_path = os.path.join(alt_root, mask_dir)
                if not os.path.isdir(img_path) or not os.path.isdir(mask_path):
                    continue
                splits[split] = BaseSegDataset.list_pairs(img_path, mask_path)
            if splits:
                return splits

        raise FileNotFoundError(
            "GTA5 data not found. Expected structure: "
            "data/gta5/images/{train,val} and data/gta5/labels/{train,val}."
        )
