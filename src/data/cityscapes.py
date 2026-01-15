"""Cityscapes dataset."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base_dataset import BaseSegDataset


class CityscapesDataset(BaseSegDataset):
    """Cityscapes semantic segmentation dataset."""

    @staticmethod
    def get_splits(root: str) -> Dict[str, List[Tuple[str, str]]]:
        # Option 1: preprocessed structure
        expected = [
            ("train", "images/train", "labels/train"),
            ("val", "images/val", "labels/val"),
            ("test", "images/test", "labels/test"),
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

        # Option 2: official Cityscapes layout (leftImg8bit/gtFine)
        cs_root = root
        left = os.path.join(cs_root, "leftImg8bit")
        fine = os.path.join(cs_root, "gtFine")
        if not (os.path.isdir(left) and os.path.isdir(fine)):
            parent = os.path.dirname(cs_root)
            if os.path.isdir(os.path.join(parent, "leftImg8bit")) and os.path.isdir(os.path.join(parent, "gtFine")):
                left = os.path.join(parent, "leftImg8bit")
                fine = os.path.join(parent, "gtFine")

        if not (os.path.isdir(left) and os.path.isdir(fine)):
            raise FileNotFoundError(
                "Cityscapes data not found. Expected either "
                "data/cityscapes/images/{train,val,test} and data/cityscapes/labels/{train,val,test} "
                "or official leftImg8bit/gtFine structure."
            )

        splits = {}
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(left, split)
            mask_dir = os.path.join(fine, split)
            images = []
            for root_dir, _, files in os.walk(img_dir):
                for name in files:
                    if name.endswith("_leftImg8bit.png"):
                        images.append(os.path.join(root_dir, name))
            images = sorted(images)
            pairs = []
            for img_path in images:
                rel = os.path.relpath(img_path, img_dir)
                base = rel.replace("_leftImg8bit.png", "")
                candidate_train = os.path.join(mask_dir, base + "_gtFine_labelTrainIds.png")
                candidate_ids = os.path.join(mask_dir, base + "_gtFine_labelIds.png")
                if os.path.exists(candidate_train):
                    mask_path = candidate_train
                elif os.path.exists(candidate_ids):
                    mask_path = candidate_ids
                else:
                    continue
                pairs.append((img_path, mask_path))
            splits[split] = pairs
        return splits
