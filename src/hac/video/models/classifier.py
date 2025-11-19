#!/usr/bin/env python3
"""Updated video/models/classifier.py with lightweight models.

Add this to your existing classifier.py
"""

import torch.nn as nn
from torchvision.models.video import (
    MC3_18_Weights,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
    mc3_18,
    mvit_v1_b,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)

MODEL_REGISTRY = {
    "r3d_18": (r3d_18, R3D_18_Weights),
    "mc3_18": (mc3_18, MC3_18_Weights),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
    "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights),
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights),
    "swin3d_t": (swin3d_t, Swin3D_T_Weights),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights),
}


class Video3DCNN(nn.Module):
    """3D CNN for video action recognition."""

    def __init__(
        self, num_classes=101, model_name="r3d_18", pretrained=True, dropout=0.3
    ):
        super().__init__()

        try:
            constructor, weight_class = MODEL_REGISTRY[model_name]
        except KeyError:
            raise ValueError(f"Unknown model: {model_name}")

        weights = weight_class.DEFAULT if pretrained else None
        self.backbone = constructor(weights=weights)

        # Determine classifier field based on model type
        if model_name in ["r3d_18", "mc3_18", "r2plus1d_18"]:
            # ResNet3D family
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(features, num_classes)

        elif model_name in [
            "mvit_v1_b",
            "mvit_v2_s",
            "swin3d_t",
            "swin3d_s",
            "swin3d_b",
        ]:
            # Transformer-based video models in torchvision 0.23+
            # Classifier is always the 'head' Linear layer
            features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(features, num_classes)

        else:
            raise ValueError(f"Unknown classifier type for model {model_name}")

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
