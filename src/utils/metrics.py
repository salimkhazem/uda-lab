"""Metrics for segmentation evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from skimage.morphology import skeletonize


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = 255) -> Dict[str, float]:
    """Compute per-class IoU and mIoU.

    Args:
        pred: Predicted labels (H, W).
        gt: Ground-truth labels (H, W).
        num_classes: Number of classes.
        ignore_index: Ignore label.

    Returns:
        Dict with per-class IoU and mIoU.
    """
    ious = []
    metrics: Dict[str, float] = {}
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = pred == cls
        gt_c = gt == cls
        if ignore_index is not None:
            valid = gt != ignore_index
            pred_c = pred_c & valid
            gt_c = gt_c & valid
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        iou = _safe_div(inter, union)
        metrics[f"iou_{cls}"] = iou
        ious.append(iou)
    metrics["miou"] = float(np.mean(ious)) if ious else 0.0
    return metrics


def dice_score(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = 255) -> Dict[str, float]:
    """Compute per-class Dice and mean Dice.

    Args:
        pred: Predicted labels (H, W).
        gt: Ground-truth labels (H, W).
        num_classes: Number of classes.
        ignore_index: Ignore label.

    Returns:
        Dict with per-class Dice and mean.
    """
    dices = []
    metrics: Dict[str, float] = {}
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = pred == cls
        gt_c = gt == cls
        if ignore_index is not None:
            valid = gt != ignore_index
            pred_c = pred_c & valid
            gt_c = gt_c & valid
        inter = (pred_c & gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        score = _safe_div(2 * inter, denom)
        metrics[f"dice_{cls}"] = score
        dices.append(score)
    metrics["mdice"] = float(np.mean(dices)) if dices else 0.0
    return metrics


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray, ignore_index: int = 255) -> float:
    """Compute pixel accuracy.

    Args:
        pred: Predicted labels.
        gt: Ground-truth labels.
        ignore_index: Ignore label.

    Returns:
        Pixel accuracy.
    """
    valid = gt != ignore_index if ignore_index is not None else np.ones_like(gt, dtype=bool)
    correct = (pred == gt) & valid
    return _safe_div(correct.sum(), valid.sum())


def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute clDice for binary masks.

    Args:
        pred: Binary prediction (H, W).
        gt: Binary ground truth (H, W).

    Returns:
        clDice score.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    tprec = _safe_div((skel_pred & gt).sum(), skel_pred.sum())
    tsens = _safe_div((skel_gt & pred).sum(), skel_gt.sum())
    return _safe_div(2 * tprec * tsens, tprec + tsens)


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = 255
) -> Dict[str, float]:
    """Compute all metrics.

    Args:
        pred: Predicted labels.
        gt: Ground truth labels.
        num_classes: Number of classes.
        ignore_index: Ignore label.

    Returns:
        Dict of metrics.
    """
    metrics = {}
    metrics.update(compute_iou(pred, gt, num_classes, ignore_index))
    metrics.update(dice_score(pred, gt, num_classes, ignore_index))
    metrics["pixel_acc"] = pixel_accuracy(pred, gt, ignore_index)
    if num_classes == 2:
        metrics["cldice"] = cldice_score(pred == 1, gt == 1)
    return metrics
