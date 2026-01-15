"""Logging utilities for TensorBoard and W&B."""

from __future__ import annotations

from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger for TensorBoard and optional W&B.

    Args:
        log_dir: TensorBoard log directory.
        use_wandb: Whether to enable W&B logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team name.
        config: Experiment configuration for W&B.
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        wandb_project: str = "",
        wandb_entity: str = "",
        config: Optional[Dict] = None,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            import wandb

            self._wandb = wandb
            self._wandb.init(project=wandb_project, entity=wandb_entity or None, config=config)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log scalar metrics.

        Args:
            metrics: Dict of metric name to value.
            step: Global step.
            prefix: Optional prefix for metric names.
        """
        for key, value in metrics.items():
            name = f"{prefix}{key}"
            self.writer.add_scalar(name, value, step)
        if self.use_wandb and self._wandb:
            self._wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        """Close the logger."""
        self.writer.close()
        if self.use_wandb and self._wandb:
            self._wandb.finish()
