"""Image I/O utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    """Read an RGB image from disk.

    Args:
        path: Image path.

    Returns:
        Image as HxWx3 numpy array.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)


def read_mask(path: str) -> np.ndarray:
    """Read a mask from disk.

    Args:
        path: Mask path.

    Returns:
        Mask as numpy array.
    """
    mask = Image.open(path)
    return np.array(mask)


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to (height, width).

    Args:
        img: Input image array.
        size: (height, width).

    Returns:
        Resized image array.
    """
    pil = Image.fromarray(img)
    pil = pil.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(pil)


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save a mask to disk.

    Args:
        mask: Mask array.
        path: Output path.
    """
    Image.fromarray(mask.astype(np.uint8)).save(path)
