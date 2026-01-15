"""HuggingFace datasets wrappers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path
from datasets import Image as HFImage, ClassLabel, load_dataset

from .base_dataset import BaseSegmentationDataset


class HFDataset(BaseSegmentationDataset):
    """Wrapper around HuggingFace datasets for segmentation."""

    IS_HF = True

    def __init__(
        self,
        root: str,
        split: str,
        img_size: Tuple[int, int],
        transform=None,
        ignore_index: int = 255,
        num_classes: int = 2,
        label_map: np.ndarray | None = None,
        hf_id: str | None = None,
        hf_config_name: str | None = None,
        image_key: str = "image",
        mask_key: str = "mask",
        split_map: Dict[str, str] | None = None,
        allow_missing_mask: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            img_size=img_size,
            transform=transform,
            ignore_index=ignore_index,
            num_classes=num_classes,
            label_map=label_map,
        )
        split_name = split_map.get(split, split) if split_map else split
        self.dataset = load_dataset(hf_id, hf_config_name, split=split_name)
        self.image_key = image_key
        self.mask_key = mask_key
        self.allow_missing_mask = allow_missing_mask
        self._paired = True
        self._pairs: List[Tuple[int, int]] = []
        self._init_pairing()

    def _init_pairing(self) -> None:
        """Handle datasets where inputs and labels are stored as separate rows."""
        if self.mask_key in self.dataset.features:
            feat = self.dataset.features[self.mask_key]
            if isinstance(feat, HFImage):
                self._paired = True
                return

        if "label" in self.dataset.features and isinstance(self.dataset.features["label"], ClassLabel):
            try:
                names = list(getattr(self.dataset.features["label"], "names", []))
            except Exception:
                names = []
            if names == ["input", "label"]:
                ds_paths = self.dataset.cast_column(self.image_key, HFImage(decode=False))
                inputs: Dict[str, int] = {}
                labels: Dict[str, int] = {}
                for i in range(len(ds_paths)):
                    ex = ds_paths[i]
                    kind = int(ex["label"])
                    path = ex[self.image_key]["path"]
                    stem = Path(path).stem
                    if kind == 0:
                        inputs[stem] = i
                    else:
                        labels[stem] = i
                common = sorted(set(inputs) & set(labels), key=lambda s: int(s) if s.isdigit() else s)
                if common:
                    self._pairs = [(inputs[k], labels[k]) for k in common]
                    self._paired = False
                    return

                input_idx = [i for i in range(len(self.dataset)) if int(self.dataset[i]["label"]) == 0]
                label_idx = [i for i in range(len(self.dataset)) if int(self.dataset[i]["label"]) == 1]
                if len(input_idx) == len(label_idx) and len(input_idx) > 0:
                    self._pairs = list(zip(input_idx, label_idx))
                    self._paired = False
                    return

        self._paired = True

    def __len__(self) -> int:
        return len(self._pairs) if not self._paired else len(self.dataset)

    def __getitem__(self, idx: int):
        if self._paired:
            sample = self.dataset[idx]
            image = np.array(sample[self.image_key])
            mask = sample.get(self.mask_key)
            if mask is None and self.allow_missing_mask:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            elif mask is None:
                raise ValueError("Missing mask in dataset sample.")
            mask = np.array(mask)
        else:
            img_idx, mask_idx = self._pairs[idx]
            image = np.array(self.dataset[img_idx][self.image_key])
            mask = np.array(self.dataset[mask_idx][self.image_key])
        mask = self._apply_label_map(mask)
        image, mask = self._apply_transform(image, mask)
        mask = self._normalize_mask(mask)
        try:
            import torch

            if isinstance(mask, torch.Tensor):
                mask = mask.long()
        except Exception:
            pass
        return {"image": image, "mask": mask, "meta": {"index": idx}}
