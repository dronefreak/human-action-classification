"""Action classification models using PyTorch and timm."""

from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """Action classifier using pretrained vision models.

    Supports both pose-based and scene-based classification.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        num_classes: int = 40,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_type: str = "image",  # "image" or "keypoints"
    ):
        """Initialize action classifier.

        Args:
            model_name: timm model name
            (e.g., "mobilenetv3_small_100", "efficientnet_b0")
            num_classes: Number of action classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights during training
            input_type: "image" for scene classification, "keypoints" for pose-based
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.input_type = input_type

        if input_type == "image":
            # Scene classification: use full pretrained model
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_classes
            )

            if freeze_backbone:
                # Freeze all except final classifier
                for name, param in self.backbone.named_parameters():
                    if (
                        "classifier" not in name
                        and "fc" not in name
                        and "head" not in name
                    ):
                        param.requires_grad = False

        else:
            # Keypoint-based: custom MLP on pose features
            # MediaPipe gives 33 keypoints x 2 coords = 66 features
            self.backbone = nn.Sequential(
                nn.Linear(66, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
               - For image: (B, 3, H, W)
               - For keypoints: (B, 66)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_type": self.input_type,
        }


class DualTaskClassifier(nn.Module):
    """
    Dual-task classifier: pose classification + action recognition.
    Mirrors the original v1 architecture but modernized.
    """

    def __init__(
        self,
        pose_classifier: Optional[ActionClassifier] = None,
        action_classifier: Optional[ActionClassifier] = None,
    ):
        """Initialize dual-task classifier.

        Args:
            pose_classifier: Model for pose classification (sitting/standing/lying)
            action_classifier: Model for action classification (40 classes)
        """
        super().__init__()

        # Pose classifier: small model, 3 classes
        if pose_classifier is None:
            self.pose_classifier = ActionClassifier(
                model_name="mobilenetv3_small_100",
                num_classes=3,
                pretrained=True,
                input_type="image",
            )
        else:
            self.pose_classifier = pose_classifier

        # Action classifier: larger model, 40 classes
        if action_classifier is None:
            self.action_classifier = ActionClassifier(
                model_name="efficientnet_b0",
                num_classes=40,
                pretrained=True,
                input_type="image",
            )
        else:
            self.action_classifier = action_classifier

    def forward(
        self, x: torch.Tensor, return_both: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through both classifiers.

        Args:
            x: Input image tensor (B, 3, H, W)
            return_both: If True, return both pose and action predictions

        Returns:
            Dictionary with 'pose' and/or 'action' logits
        """
        results = {}

        if return_both:
            results["pose"] = self.pose_classifier(x)
            results["action"] = self.action_classifier(x)
        else:
            results["action"] = self.action_classifier(x)

        return results


def create_model(
    model_type: str = "action",
    model_name: str = "mobilenetv3_small_100",
    num_classes: int = 40,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create models.

    Args:
        model_type: "action", "pose", "dual", or "keypoint"
        model_name: timm model name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments

    Returns:
        PyTorch model
    """
    if model_type == "action":
        return ActionClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            input_type="image",
            freeze_backbone=freeze_backbone,
            **kwargs,
        )

    elif model_type == "pose":
        return ActionClassifier(
            model_name=model_name,
            num_classes=3,  # sitting, standing, lying
            pretrained=pretrained,
            input_type="image",
            freeze_backbone=freeze_backbone,
            **kwargs,
        )

    elif model_type == "keypoint":
        return ActionClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            input_type="keypoints",
            freeze_backbone=freeze_backbone,
            **kwargs,
        )

    elif model_type == "dual":
        return DualTaskClassifier(**kwargs)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
