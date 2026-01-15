"""Metric tracking utilities."""

from __future__ import annotations

from typing import Dict


class AverageMeter:
    """Track running averages of a metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset meter state."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Update meter with new value.

        Args:
            value: Metric value.
            n: Number of samples.
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """Track a set of metrics over time."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        """Update multiple metrics.

        Args:
            metrics: Dict of metric name to value.
            n: Number of samples for weighting.
        """
        for key, value in metrics.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter()
            self.meters[key].update(float(value), n=n)

    def average(self) -> Dict[str, float]:
        """Return average values."""
        return {k: v.avg for k, v in self.meters.items()}
