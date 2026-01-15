"""Distributed training utilities."""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> None:
    """Initialize torch.distributed if available.

    Args:
        backend: Distributed backend.
    """
    if dist.is_available() and not dist.is_initialized():
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(rank % torch.cuda.device_count())


def is_distributed() -> bool:
    """Return True if distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """Return True if current process is rank 0."""
    return not is_distributed() or dist.get_rank() == 0


def get_world_size() -> int:
    """Return world size."""
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    """Return rank."""
    return dist.get_rank() if is_distributed() else 0


def reduce_dict(values: Dict[str, float]) -> Dict[str, float]:
    """Reduce metric dict across processes.

    Args:
        values: Dict of float metrics.

    Returns:
        Reduced dict.
    """
    if not is_distributed():
        return values
    keys = sorted(values.keys())
    tensor = torch.tensor([values[k] for k in keys], device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}
