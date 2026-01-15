"""Training hooks."""

from __future__ import annotations

from typing import Dict

import torch


class CheckpointSaver:
    """Save best checkpoint based on a metric."""

    def __init__(self, metric: str = "miou", maximize: bool = True) -> None:
        self.metric = metric
        self.maximize = maximize
        self.best = -1e9 if maximize else 1e9

    def update(self, metrics: Dict[str, float], model: torch.nn.Module, path: str) -> bool:
        """Update and save if improved.

        Args:
            metrics: Current metrics dict.
            model: Model to save.
            path: Output checkpoint path.

        Returns:
            True if checkpoint saved.
        """
        value = metrics.get(self.metric, None)
        if value is None:
            return False
        improved = value > self.best if self.maximize else value < self.best
        if improved:
            self.best = value
            torch.save({"state_dict": model.state_dict(), "metric": value}, path)
            return True
        return False
