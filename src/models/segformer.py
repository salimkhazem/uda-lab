"""SegFormer wrapper using transformers."""

from __future__ import annotations

import torch
from torch import nn


class SegFormer(nn.Module):
    """SegFormer segmentation model wrapper."""

    def __init__(self, num_classes: int, backbone: str = "mit_b0", pretrained: bool = True) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        model_map = {
            "mit_b0": "nvidia/mit-b0",
            "mit_b1": "nvidia/mit-b1",
            "mit_b2": "nvidia/mit-b2",
        }
        model_id = model_map.get(backbone, "nvidia/mit-b0")
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            cfg = SegformerConfig.from_pretrained(model_id, num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder features for alignment."""
        outputs = self.model.segformer(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states[-1]
