"""UDA trainer."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from ..losses.consistency import ConsistencyLoss
from ..losses.domain_adv import DomainAdversarialLoss
from ..losses.loss_factory import build_losses
from ..losses.topology import topology_consistency_loss
from ..utils.distributed import reduce_dict, is_main_process
from ..utils.meters import MetricTracker
from ..utils.metrics import compute_metrics
from .hooks import CheckpointSaver
from .pseudo_labeler import make_pseudo_labels


def strong_augment(x: torch.Tensor) -> torch.Tensor:
    """Simple strong augmentation for tensors."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    noise = torch.randn_like(x) * 0.05
    return torch.clamp(x + noise, 0.0, 1.0)


class UDATrainer:
    """Trainer for UDA segmentation."""

    def __init__(self, cfg: Dict, model: torch.nn.Module, loaders: Dict, logger, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model
        self.loaders = loaders
        self.logger = logger
        self.device = device
        self.losses = build_losses(cfg)
        self.scaler = GradScaler("cuda", enabled=cfg["train"]["amp"])
        self.consistency_loss = ConsistencyLoss(use_probs=True)
        self.adv_loss = None

    def _init_adv(self, features: torch.Tensor) -> None:
        if self.adv_loss is None:
            self.adv_loss = DomainAdversarialLoss(in_channels=features.shape[1]).to(features.device)

    def train_one_epoch(self, optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricTracker()
        src_loader = self.loaders["train"]
        tgt_loader = self.loaders["target"]
        tgt_iter = iter(tgt_loader)

        grad_accum = max(int(self.cfg["train"]["grad_accum"]), 1)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(src_loader, desc=f"uda train {epoch}", leave=False)):
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            src_img = batch["image"].to(self.device)
            src_mask = batch["mask"].to(self.device)
            tgt_img = tgt_batch["image"].to(self.device)

            tgt_img_strong = strong_augment(tgt_img) if self.cfg["uda"]["strong_aug"] else tgt_img

            with autocast(device_type="cuda", enabled=self.cfg["train"]["amp"]):
                src_logits = self.model(src_img)
                sup_loss = self.losses["ce_dice"](src_logits, src_mask)

                tgt_logits = self.model(tgt_img)
                tgt_probs = torch.softmax(tgt_logits, dim=1)
                pseudo, valid = make_pseudo_labels(
                    tgt_probs,
                    threshold=self.cfg["uda"]["pseudo_threshold"],
                    pseudo_min_size=self.cfg["uda"]["pseudo_min_size"],
                    morph=self.cfg["uda"]["pseudo_morph"],
                    skeleton_filter=self.cfg["uda"]["pseudo_skeleton_filter"],
                )
                if self.cfg["uda"]["use_pseudo_filter"]:
                    pseudo_filtered = pseudo.clone()
                    pseudo_filtered[~valid] = self.cfg["dataset"]["ignore_index"]
                    pseudo_loss = self.losses["ce_dice"](tgt_logits, pseudo_filtered)
                else:
                    pseudo_loss = self.losses["ce_dice"](tgt_logits, pseudo)

                topo_loss = torch.tensor(0.0, device=self.device)
                if self.cfg["uda"]["use_topo_loss"]:
                    pseudo_onehot = torch.nn.functional.one_hot(
                        pseudo, num_classes=self.cfg["dataset"]["num_classes"]
                    ).permute(0, 3, 1, 2)
                    topo_loss = self.losses["topology"](tgt_probs, pseudo_onehot.float())

                cons_loss = torch.tensor(0.0, device=self.device)
                if self.cfg["uda"]["consistency_weight"] > 0:
                    tgt_logits_strong = self.model(tgt_img_strong)
                    cons_loss = self.consistency_loss(tgt_logits, tgt_logits_strong)

                adv_loss = torch.tensor(0.0, device=self.device)
                if self.cfg["uda"]["use_adv"]:
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    feats = model_ref.get_features(torch.cat([src_img, tgt_img], dim=0))
                    self._init_adv(feats)
                    domain_labels = torch.cat(
                        [
                            torch.zeros(src_img.size(0), device=self.device),
                            torch.ones(tgt_img.size(0), device=self.device),
                        ],
                        dim=0,
                    )
                    adv_loss = self.adv_loss(feats, domain_labels)

                total = (
                    sup_loss
                    + self.cfg["uda"]["consistency_weight"] * cons_loss
                    + self.cfg["uda"]["topo_weight"] * topo_loss
                    + self.cfg["uda"]["adv_weight"] * adv_loss
                    + pseudo_loss
                )

            self.scaler.scale(total).backward()
            if (step + 1) % grad_accum == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tracker.update(
                {
                    "loss": float(total.item()),
                    "sup_loss": float(sup_loss.item()),
                    "pseudo_loss": float(pseudo_loss.item()),
                    "topo_loss": float(topo_loss.item()),
                },
                n=src_img.size(0),
            )

        return reduce_dict(tracker.average())

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict[str, float]:
        """Evaluate on validation split."""
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
        return reduce_dict(tracker.average())

    def fit(self, optimizer, scheduler, out_dir: str) -> Dict[str, List[float]]:
        """Fit for multiple epochs."""
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
