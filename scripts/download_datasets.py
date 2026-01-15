#!/usr/bin/env python
"""Dataset download instructions and validators."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


INSTRUCTIONS = {
    "drive": "Use the DRIVE dataset from https://drive.grand-challenge.org/. "
    "After download, place files under data/drive/images/{train,val,test} "
    "and data/drive/masks/{train,val,test}.",
    "stare": "STARE requires manual download. Place files under data/stare/images/{train,val,test} "
    "and data/stare/masks/{train,val,test}.",
    "chase": "CHASEDB1 requires manual download. Place files under data/chase/images/{train,val,test} "
    "and data/chase/masks/{train,val,test}.",
    "deepglobe": "Download from Kaggle (DeepGlobe Road Extraction). Place files under "
    "data/deepglobe/images/{train,val,test} and data/deepglobe/masks/{train,val,test}.",
    "spacenet": "Download SpaceNet Roads from AWS/Open data. Place files under "
    "data/spacenet/images/{train,val,test} and data/spacenet/masks/{train,val,test}.",
    "gta5": "Download GTA5 labels from https://download.visinf.tu-darmstadt.de/data/from_games/. "
    "Place under data/gta5/images/{train,val} and data/gta5/labels/{train,val}.",
    "cityscapes": "Cityscapes requires registration: https://www.cityscapes-dataset.com/. "
    "Place under data/cityscapes/images/{train,val,test} and data/cityscapes/labels/{train,val,test}.",
    "gta5_cityscapes": "Download GTA5 and Cityscapes datasets. Place GTA5 under data/gta5/images/{train,val} "
    "and data/gta5/labels/{train,val}. Place Cityscapes under data/cityscapes/images/{train,val,test} and "
    "data/cityscapes/labels/{train,val,test}.",
    "ssdd": "SSDD requires manual download. Place under data/ssdd/images/{train,val,test} "
    "and data/ssdd/masks/{train,val,test}.",
    "kvasir": "Kvasir can be used from HuggingFace. Provide hf_id via config or download manually. "
    "If local, place under data/kvasir/images/{train,val,test} and data/kvasir/masks/{train,val,test}.",
    "retina": "Retina dataset can be used from HuggingFace. Provide hf_id via config or download manually. "
    "If local, place under data/retina/images/{train,val,test} and data/retina/masks/{train,val,test}.",
}


def validate_paths(root: str, folders: list[str]) -> None:
    missing = []
    for folder in folders:
        path = Path(root) / folder
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing paths:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All expected paths found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    name = args.dataset.lower()
    print(INSTRUCTIONS.get(name, "No instructions available for this dataset."))
    if name in {"drive", "stare", "chase", "deepglobe", "spacenet", "ssdd"}:
        validate_paths(f"data/{name}", ["images/train", "masks/train"])
    if name in {"kvasir", "retina"}:
        validate_paths(f"data/{name}", ["images/train", "masks/train"])
    if name == "gta5":
        validate_paths("data/gta5", ["images/train", "labels/train"])
    if name == "cityscapes":
        validate_paths("data/cityscapes", ["images/train", "labels/train"])
    if name == "gta5_cityscapes":
        validate_paths("data/gta5", ["images/train", "labels/train"])
        validate_paths("data/cityscapes", ["images/train", "labels/train"])


if __name__ == "__main__":
    main()
