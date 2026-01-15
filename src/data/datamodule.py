"""Dataset builders and dataloaders."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .base_dataset import BaseSegDataset
from .chase import CHASEDataset
from .cityscapes import CityscapesDataset
from .deepglobe_roads import DeepGlobeRoadsDataset
from .drive import DRIVEDataset
from .gta5 import GTA5Dataset
from .hf_wrappers import HFDataset
from .spacenet_roads import SpaceNetRoadsDataset
from .ssdd import SSDDDataset
from .stare import STAREDataset
from .transforms import build_transforms, build_weak_strong
from ..trainers.diffusion_aug import augment_pairs, load_manifest
from ..utils.distributed import is_distributed
from ..utils.seed import seed_worker


DATASET_REGISTRY = {
    "drive": DRIVEDataset,
    "stare": STAREDataset,
    "chase": CHASEDataset,
    "deepglobe_roads": DeepGlobeRoadsDataset,
    "spacenet_roads": SpaceNetRoadsDataset,
    "gta5": GTA5Dataset,
    "cityscapes": CityscapesDataset,
    "ssdd": SSDDDataset,
}


def cityscapes_label_map(ignore_index: int = 255) -> np.ndarray:
    """Map Cityscapes/GTA5 label IDs to train IDs."""
    mapping = {i: ignore_index for i in range(256)}
    train_id = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18,
    }
    mapping.update(train_id)
    lut = [mapping[i] for i in range(256)]
    return np.array(lut, dtype=np.uint8)


def _resolve_limit(limit_cfg, split: str):
    if limit_cfg is None:
        return None
    if isinstance(limit_cfg, int):
        return limit_cfg
    if isinstance(limit_cfg, dict):
        return limit_cfg.get(split)
    return None


def _apply_sample_limit(dataset, limit: int, seed: int) -> None:
    if not limit or limit <= 0:
        return
    if hasattr(dataset, "samples"):
        indices = list(range(len(dataset.samples)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[: min(limit, len(indices))]
        dataset.samples = [dataset.samples[i] for i in indices]
        return
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "select"):
        ds_len = len(dataset.dataset)
        n = min(limit, ds_len)
        dataset.dataset = dataset.dataset.shuffle(seed=seed).select(range(n))


def build_dataset(
    name: str,
    root: str,
    img_size: Tuple[int, int],
    split: str,
    ignore_index: int,
    num_classes: int,
    is_train: bool,
    dataset_cfg: Dict | None = None,
) -> BaseSegDataset:
    """Build dataset instance."""
    transform = build_transforms(img_size, is_train=is_train)
    label_map = None
    if name in {"cityscapes", "gta5"}:
        label_map = cityscapes_label_map(ignore_index)
    if dataset_cfg and dataset_cfg.get("label_map"):
        label_map = np.array(dataset_cfg["label_map"], dtype=np.uint8)

    if dataset_cfg and dataset_cfg.get("hf_id"):
        split_map = (dataset_cfg or {}).get("hf_splits", {})
        hf_split = split_map.get(split, split)
        return HFDataset(
            root=root,
            split=hf_split,
            img_size=img_size,
            transform=transform,
            ignore_index=ignore_index,
            num_classes=num_classes,
            label_map=label_map,
            hf_id=dataset_cfg.get("hf_id"),
            hf_config_name=dataset_cfg.get("hf_config_name"),
            image_key=dataset_cfg.get("hf_image_key", "image"),
            mask_key=dataset_cfg.get("hf_mask_key", "mask"),
            split_map=split_map,
            allow_missing_mask=dataset_cfg.get("allow_missing_mask", False),
        )

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    splits = DATASET_REGISTRY[name].get_splits(root)
    if split not in splits:
        raise ValueError(f"Split {split} not found for dataset {name}")
    return DATASET_REGISTRY[name](
        root=root,
        samples=splits[split],
        img_size=img_size,
        transform=transform,
        ignore_index=ignore_index,
        num_classes=num_classes,
        label_map=label_map,
    )


def build_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    """Build train/val dataloaders."""
    img_size = tuple(cfg["dataset"]["crop_size"])
    train_ds = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        img_size,
        cfg["dataset"]["train_split"],
        cfg["dataset"]["ignore_index"],
        cfg["dataset"]["num_classes"],
        True,
        cfg["dataset"],
    )
    train_limit = _resolve_limit(cfg["dataset"].get("max_samples"), cfg["dataset"]["train_split"])
    _apply_sample_limit(train_ds, train_limit, cfg["seed"])
    if cfg.get("uda", {}).get("use_diffusion_aug") and cfg["uda"].get("diffusion_manifest"):
        manifest = load_manifest(cfg["uda"]["diffusion_manifest"])
        train_ds.samples = augment_pairs(train_ds.samples, manifest)
        ratio = float(cfg["uda"].get("diffusion_ratio", 1.0))
        if ratio < 1.0:
            n_total = len(train_ds.samples)
            n_keep = int(n_total * ratio)
            rng = random.Random(cfg["seed"])
            indices = list(range(n_total))
            rng.shuffle(indices)
            indices = indices[: max(n_keep, 1)]
            train_ds.samples = [train_ds.samples[i] for i in indices]
    val_ds = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        img_size,
        cfg["dataset"]["val_split"],
        cfg["dataset"]["ignore_index"],
        cfg["dataset"]["num_classes"],
        False,
        cfg["dataset"],
    )
    val_limit = _resolve_limit(cfg["dataset"].get("max_samples"), cfg["dataset"]["val_split"])
    _apply_sample_limit(val_ds, val_limit, cfg["seed"] + 1)

    train_sampler = DistributedSampler(train_ds) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return {"train": train_loader, "val": val_loader}


def build_uda_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    """Build UDA source/target loaders."""
    loaders = build_dataloaders(cfg)
    img_size = tuple(cfg["dataset"]["crop_size"])
    target_cfg = dict(cfg["dataset"])
    target_name = cfg["dataset"].get("target_name") or cfg["dataset"]["name"]
    target_root = cfg["dataset"].get("target_root") or cfg["dataset"]["root"]
    target_cfg["name"] = target_name
    target_cfg["root"] = target_root
    if cfg["dataset"].get("target_hf_id"):
        target_cfg["hf_id"] = cfg["dataset"]["target_hf_id"]
    if cfg["dataset"].get("target_hf_splits"):
        target_cfg["hf_splits"] = cfg["dataset"]["target_hf_splits"]
    target_ds = build_dataset(
        target_cfg["name"],
        target_cfg["root"],
        img_size,
        cfg["dataset"].get("target_split", "train"),
        target_cfg["ignore_index"],
        target_cfg["num_classes"],
        True,
        target_cfg,
    )
    target_split = cfg["dataset"].get("target_split", "train")
    target_limit_cfg = cfg["dataset"].get("target_max_samples", cfg["dataset"].get("max_samples"))
    target_limit = _resolve_limit(target_limit_cfg, target_split)
    _apply_sample_limit(target_ds, target_limit, cfg["seed"] + 2)
    target_sampler = DistributedSampler(target_ds) if is_distributed() else None
    target_loader = DataLoader(
        target_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=target_sampler is None,
        sampler=target_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    loaders["target"] = target_loader
    return loaders


def build_weak_strong_transforms(cfg: Dict):
    """Build weak/strong transforms for consistency."""
    img_size = tuple(cfg["dataset"]["crop_size"])
    return build_weak_strong(img_size)
