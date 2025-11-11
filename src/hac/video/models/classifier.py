#!/usr/bin/env python3
"""Updated video/models/classifier.py with lightweight models.

Add this to your existing classifier.py
"""

import torch.nn as nn
from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18


class Video3DCNN(nn.Module):
    """3D CNN for video action recognition."""

    def __init__(
        self, num_classes=101, model_name="r3d_18", pretrained=True, dropout=0.3
    ):
        super().__init__()

        # Check if it's a lightweight model
        if model_name.startswith("r3d_") and model_name not in [
            "r3d_18",
            "r2plus1d_18",
        ]:
            # Lightweight model from scratch
            from hac.video.models.models import create_lightweight_model

            self.backbone = create_lightweight_model(
                model_name=model_name, num_classes=num_classes, dropout=dropout
            )
            self.is_lightweight = True

        elif model_name == "tiny_c3d":
            # Tiny C3D model
            from hac.video.models.models import TinyC3D

            self.backbone = TinyC3D(num_classes=num_classes, dropout=dropout)
            self.is_lightweight = True

        else:
            # Standard pretrained models
            if model_name == "r3d_18":
                self.backbone = r3d_18(pretrained=pretrained)
            elif model_name == "mc3_18":
                self.backbone = mc3_18(pretrained=pretrained)
            elif model_name == "r2plus1d_18":
                self.backbone = r2plus1d_18(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Replace final FC layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            self.is_lightweight = False

        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - batch of video clips
        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)

    def get_config(self):
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "type": "3dcnn_lightweight" if self.is_lightweight else "3dcnn",
            "pretrained": not self.is_lightweight,
        }
