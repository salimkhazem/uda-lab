#!/usr/bin/env python
"""Prediction script for arbitrary images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.model_factory import create_model
from src.utils.config import load_config
from src.utils.image import save_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, model=args.model, opts=args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in Path(args.input_dir).glob("**/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    with torch.no_grad():
        for img_path in tqdm(images, desc="predict"):
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            save_mask(pred.astype(np.uint8), str(out_dir / f"{img_path.stem}.png"))


if __name__ == "__main__":
    main()
