"""Unit tests for topology loss."""

from __future__ import annotations

import torch

from src.losses.topology import soft_skeletonize, cldice_loss


def test_soft_skeletonize_shape() -> None:
    prob = torch.rand(2, 1, 64, 64)
    skel = soft_skeletonize(prob, iters=5)
    assert skel.shape == prob.shape


def test_cldice_gradients() -> None:
    pred = torch.rand(1, 1, 32, 32, requires_grad=True)
    gt = (torch.rand(1, 1, 32, 32) > 0.5).float()
    loss = cldice_loss(pred, gt, iters=5)
    loss.backward()
    assert pred.grad is not None
