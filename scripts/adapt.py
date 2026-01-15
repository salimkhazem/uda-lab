#!/usr/bin/env python
"""UDA adaptation script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datamodule import build_uda_dataloaders
from src.models.model_factory import create_model
from src.trainers.uda_trainer import UDATrainer
from src.utils.config import load_config
from src.utils.distributed import init_distributed, is_distributed, is_main_process
from src.utils.io import ensure_dir, get_env_snapshot, get_git_hash, get_hardware_info, save_json, save_yaml
from src.utils.logger import Logger
from src.utils.seed import set_seed
from src.utils.plotting import plot_curves
from src.utils.image import save_mask
from src.utils.metrics import compute_metrics


def build_scheduler(optimizer, cfg):
    if cfg["train"]["scheduler"] == "poly":
        def lr_lambda(epoch):
            return (1 - epoch / max(cfg["train"]["epochs"], 1)) ** 0.9

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_tb", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    init_distributed()
    cfg = load_config(args.config, dataset=args.dataset, model=args.model, opts=args.opts)
    cfg["seed"] = args.seed
    if args.resume:
        cfg["train"]["resume"] = args.resume
    cfg["logging"]["log_tb"] = args.log_tb or cfg["logging"]["log_tb"]
    cfg["logging"]["log_wandb"] = args.log_wandb or cfg["logging"]["log_wandb"]

    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["exp"]["output_dir"]) / cfg["exp"]["name"] / args.dataset / args.model / str(cfg["seed"])
    if is_main_process():
        ensure_dir(out_dir)
        save_yaml(cfg, out_dir / "config.yaml")
        save_json(get_env_snapshot(), out_dir / "env.json")
        save_json(get_hardware_info(), out_dir / "hardware.json")
        with open(out_dir / "git_hash.txt", "w", encoding="utf-8") as f:
            f.write(get_git_hash())

    loaders = build_uda_dataloaders(cfg)
    model = create_model(cfg).to(device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if cfg["train"]["resume"]:
        checkpoint = torch.load(cfg["train"]["resume"], map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = build_scheduler(optimizer, cfg)
    logger = Logger(
        log_dir=str(out_dir / "logs"),
        use_wandb=cfg["logging"]["log_wandb"],
        wandb_project=cfg["exp"]["wandb_project"],
        wandb_entity=cfg["exp"]["wandb_entity"],
        config=cfg,
    )

    trainer = UDATrainer(cfg, model, loaders, logger, device)
    history = trainer.fit(optimizer, scheduler, str(out_dir))

    if is_main_process():
        save_json({"history": history}, out_dir / "metrics.json")
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_curves(history, str(fig_dir / "training_curves.png"))

        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loaders["target"]):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                gts = masks.cpu().numpy()
                for i, (pred, gt) in enumerate(zip(preds, gts)):
                    metrics.append(compute_metrics(pred, gt, cfg["dataset"]["num_classes"], cfg["dataset"]["ignore_index"]))
                    save_mask(pred.astype("uint8"), str(pred_dir / f"{batch_idx}_{i}.png"))
        if metrics:
            avg = {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in metrics[0]}
            save_json(avg, out_dir / "metrics_eval.json")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, out_dir / "last.ckpt")
    logger.close()


if __name__ == "__main__":
    main()
