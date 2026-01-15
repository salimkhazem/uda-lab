#!/usr/bin/env python
"""Prepare dataset splits by copying or symlinking files."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def list_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    images = sorted([p for p in Path(images_dir).glob("*") if p.suffix.lower() in exts])
    pairs = []
    for img_path in images:
        base = img_path.stem
        mask = None
        for ext in exts:
            candidate = Path(masks_dir) / f"{base}{ext}"
            if candidate.exists():
                mask = candidate
                break
        if mask is None:
            continue
        pairs.append((str(img_path), str(mask)))
    return pairs


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test : n_test + n_val]
    train = pairs[n_test + n_val :]
    return train, val, test


def link_or_copy(src: str, dst: str, copy: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        if os.path.exists(dst):
            return
        os.symlink(os.path.abspath(src), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    pairs = list_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check paths.")
    train, val, test = split_pairs(pairs, args.val_ratio, args.test_ratio, args.seed)

    for split, items in [("train", train), ("val", val), ("test", test)]:
        for img, mask in items:
            img_name = os.path.basename(img)
            mask_name = os.path.basename(mask)
            link_or_copy(img, f"{args.out_dir}/images/{split}/{img_name}", args.copy)
            link_or_copy(mask, f"{args.out_dir}/masks/{split}/{mask_name}", args.copy)

    print(f"Prepared splits under {args.out_dir}")


if __name__ == "__main__":
    main()
