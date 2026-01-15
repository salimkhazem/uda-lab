"""Pseudo-labeling utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.morphology import binary_closing, binary_opening, remove_small_objects


def make_pseudo_labels(
    probs: torch.Tensor,
    threshold: float = 0.9,
    pseudo_min_size: int = 100,
    morph: bool = True,
    skeleton_filter: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pseudo-labels with confidence thresholding.

    Args:
        probs: Softmax probabilities [B, C, H, W].
        threshold: Confidence threshold.
        pseudo_min_size: Minimum object size to keep.
        morph: Whether to apply morphological filtering.
        skeleton_filter: Whether to discard noisy pseudo labels (simple heuristic).

    Returns:
        Tuple of (pseudo_labels, mask of valid pixels).
    """
    conf, pseudo = probs.max(dim=1)
    valid = conf >= threshold
    pseudo_np = pseudo.cpu().numpy()
    valid_np = valid.cpu().numpy()

    if morph:
        for b in range(pseudo_np.shape[0]):
            mask = valid_np[b]
            if mask.any():
                keep = remove_small_objects(mask.astype(bool), min_size=pseudo_min_size)
                keep = binary_opening(keep)
                keep = binary_closing(keep)
                valid_np[b] = keep

    if skeleton_filter:
        # Heuristic: require minimum fraction of confident pixels
        for b in range(valid_np.shape[0]):
            if valid_np[b].mean() < 0.01:
                valid_np[b] = False

    pseudo = torch.from_numpy(pseudo_np).to(probs.device)
    valid = torch.from_numpy(valid_np).to(probs.device)
    return pseudo, valid
