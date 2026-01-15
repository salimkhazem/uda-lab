"""Diffusion augmentation utilities."""

from __future__ import annotations

import json
from typing import List, Tuple


def load_manifest(path: str) -> dict:
    """Load diffusion augmentation manifest."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def augment_pairs(pairs: List[Tuple[str, str]], manifest: dict) -> List[Tuple[str, str]]:
    """Append diffusion-augmented image pairs based on manifest."""
    augmented = []
    for img_path, mask_path in pairs:
        if img_path in manifest:
            augmented.append((manifest[img_path], mask_path))
    return pairs + augmented
