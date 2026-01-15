#!/usr/bin/env python
"""Evaluation script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datamodule import build_dataloaders, build_uda_dataloaders
from src.models.model_factory import create_model
from src.utils.config import load_config
from src.utils.image import save_mask
from src.utils.metrics import compute_metrics
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--use_target", action="store_true")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, model=args.model, opts=args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_uda_dataloaders(cfg) if args.use_target else build_dataloaders(cfg)
    loader = loaders["target"] if args.use_target else loaders["val"]

    model = create_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    if args.save_preds:
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    metrics_accum = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.cpu().numpy()
            for i, (pred, gt) in enumerate(zip(preds, gts)):
                metrics = compute_metrics(pred, gt, cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"])
                metrics_accum.append(metrics)
                if args.save_preds:
                    save_mask(pred.astype(np.uint8), str(out_dir / "predictions" / f"{batch_idx}_{i}.png"))

    # Average metrics
    avg = {}
    for key in metrics_accum[0].keys():
        avg[key] = float(np.mean([m[key] for m in metrics_accum]))
    save_json(avg, out_dir / "metrics_eval.json")
    print(avg)


if __name__ == "__main__":
    main()
