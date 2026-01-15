"""Loss factory utilities."""

from __future__ import annotations

from typing import Dict

from .ce_dice import CEDiceLoss
from .topology import TopologyLoss


def build_losses(cfg: Dict) -> Dict:
    """Build standard loss objects.

    Args:
        cfg: Config dict.

    Returns:
        Dict of loss modules.
    """
    losses = {
        "ce_dice": CEDiceLoss(
            num_classes=cfg["dataset"]["num_classes"],
            ignore_index=cfg["dataset"]["ignore_index"],
            ce_weight=cfg["loss"]["ce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
        ),
        "topology": TopologyLoss(iters=cfg["uda"].get("topology_iters", 10)),
    }
    return losses
