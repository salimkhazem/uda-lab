#!/usr/bin/env python
"""Verify that images and masks have consistent shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="")
    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root / "images/train"
    mask_dir = root / "masks/train"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Expected images/train and masks/train under root.")

    mismatches = 0
    for img_path in img_dir.glob("*"):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if img.size != mask.size:
            mismatches += 1
            print(f"Mismatch: {img_path} {img.size} vs {mask_path} {mask.size}")

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for src, aug in manifest.items():
            if not Path(src).exists() or not Path(aug).exists():
                print(f"Missing in manifest: {src} -> {aug}")

    print(f"Total mismatches: {mismatches}")


if __name__ == "__main__":
    main()
