"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random
import warnings
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Global random seed.
        deterministic: Whether to enable deterministic algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        warnings.filterwarnings(
            "ignore",
            message=".*nll_loss2d_forward_out_cuda_template does not have a deterministic implementation.*",
            category=UserWarning,
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Initialize dataloader worker seeds.

    Args:
        worker_id: PyTorch DataLoader worker id.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
