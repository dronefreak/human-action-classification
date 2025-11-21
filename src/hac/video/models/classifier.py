#!/usr/bin/env python3
"""Updated video/models/classifier.py with lightweight models."""

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
            # Transformer-based video models
            # Head structure varies by model and torchvision version:
            # - Some have: Sequential(Dropout, Linear)
            # - Some have: Linear directly

            if isinstance(self.backbone.head, nn.Sequential):
                # Head is Sequential - extract features from last Linear layer
                last_layer = list(self.backbone.head.children())[-1]
                if isinstance(last_layer, nn.Linear):
                    features = last_layer.in_features
                else:
                    raise ValueError(
                        f"Expected Linear layer in head, got {type(last_layer)}"
                    )
                # Replace entire Sequential with single Linear
                self.backbone.head = nn.Linear(features, num_classes)

            elif isinstance(self.backbone.head, nn.Linear):
                # Head is directly a Linear layer (like Swin3D in torchvision 0.23+)
                features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(features, num_classes)

            else:
                raise ValueError(
                    f"Unexpected head type for {model_name}: {type(self.backbone.head)}"
                )

        else:
            raise ValueError(f"Unknown classifier type for model {model_name}")

        self.is_lightweight = False

        self.model_name = model_name
        self.num_classes = num_classes

    def freeze_backbone(self):
        """Freeze all backbone parameters except the final classifier.

        This is useful for transfer learning on small datasets. Only the final fc/head
        layer will be trainable.
        """
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the final classifier layer
        if self.model_name in ["r3d_18", "mc3_18", "r2plus1d_18"]:
            # ResNet3D family - unfreeze fc layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name in [
            "mvit_v1_b",
            "mvit_v2_s",
            "swin3d_t",
            "swin3d_s",
            "swin3d_b",
        ]:
            # Transformer-based - unfreeze head
            for param in self.backbone.head.parameters():
                param.requires_grad = True

        print(f"✓ Backbone frozen. Only training final classifier layer.")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} parameters")

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
