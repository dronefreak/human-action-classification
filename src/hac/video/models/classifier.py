"""3D CNN models for video action recognition."""

import torch.nn as nn
from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18


class Video3DCNN(nn.Module):
    """3D CNN for video action recognition."""

    def __init__(self, num_classes=101, model_name="r3d_18", pretrained=True):
        super().__init__()

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
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes)
        )

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
            "type": "3dcnn",
        }
