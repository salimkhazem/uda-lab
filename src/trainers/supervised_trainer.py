"""Supervised trainer."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from ..losses.loss_factory import build_losses
from ..utils.meters import MetricTracker
from ..utils.metrics import compute_metrics
from .hooks import CheckpointSaver
from ..utils.distributed import reduce_dict, is_main_process


class SupervisedTrainer:
    """Trainer for supervised segmentation."""

    def __init__(self, cfg: Dict, model: torch.nn.Module, loaders: Dict, logger, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model
        self.loaders = loaders
        self.logger = logger
        self.device = device
        self.losses = build_losses(cfg)
        self.scaler = GradScaler("cuda", enabled=cfg["train"]["amp"])

    def train_one_epoch(self, optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricTracker()
        loader = self.loaders["train"]
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"train {epoch}", leave=False)):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            with autocast(device_type="cuda", enabled=self.cfg["train"]["amp"]):
                logits = self.model(images)
                loss = self.losses["ce_dice"](logits, masks)
            self.scaler.scale(loss).backward()
            if (step + 1) % max(int(self.cfg["train"]["grad_accum"]), 1) == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
            tracker.update({"loss": float(loss.item())}, n=images.size(0))
        metrics = reduce_dict(tracker.average())
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        loader = self.loaders[split]
        tracker = MetricTracker()
        for batch in tqdm(loader, desc=f"eval {split}", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.cpu().numpy()
            for pred, gt in zip(preds, gts):
                metrics = compute_metrics(pred, gt, self.cfg["dataset"]["num_classes"], self.cfg["dataset"]["ignore_index"])
                tracker.update(metrics, n=1)
        metrics = reduce_dict(tracker.average())
        return metrics

    def fit(self, optimizer, scheduler, out_dir: str) -> Dict[str, List[float]]:
        """Fit model for multiple epochs."""
        history: Dict[str, List[float]] = {"loss": [], "miou": []}
        saver = CheckpointSaver(metric="miou", maximize=True)
        for epoch in range(self.cfg["train"]["epochs"]):
            train_metrics = self.train_one_epoch(optimizer, epoch)
            if scheduler is not None:
                scheduler.step()
            if (epoch + 1) % self.cfg["train"]["eval_interval"] == 0:
                val_metrics = self.evaluate("val")
                if is_main_process():
                    self.logger.log_metrics(train_metrics, epoch, prefix="train/")
                    self.logger.log_metrics(val_metrics, epoch, prefix="val/")
                    history["loss"].append(train_metrics.get("loss", 0.0))
                    history["miou"].append(val_metrics.get("miou", 0.0))
                    saver.update(val_metrics, self.model, f"{out_dir}/best.ckpt")
        return history
