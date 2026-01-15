"""Plotting utilities for training curves and qualitative grids."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _colorize_mask(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Colorize a mask for visualization."""
    if num_classes <= 2:
        return (mask > 0).astype(np.uint8) * 255
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_classes))[:, :3]
    colored = colors[mask]
    return (colored * 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a mask on top of an image."""
    img = image.astype(np.float32)
    m = _colorize_mask(mask)
    if m.ndim == 2:
        m = np.repeat(m[..., None], 3, axis=-1)
    return np.clip((1 - alpha) * img + alpha * m, 0, 255).astype(np.uint8)


def plot_curves(history: dict, out_path: str) -> None:
    """Plot training curves from history dict."""
    plt.figure(figsize=(8, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_comparison_grid(
    images: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    base_preds: Sequence[np.ndarray],
    full_preds: Sequence[np.ndarray],
    out_path: str,
    max_items: int = 8,
) -> None:
    """Save a qualitative comparison grid."""
    n = min(len(images), max_items)
    plt.figure(figsize=(4 * n, 12))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.subplot(3, n, n + i + 1)
        plt.imshow(_colorize_mask(gts[i]))
        plt.axis("off")
        plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(_colorize_mask(full_preds[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
