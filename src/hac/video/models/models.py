#!/usr/bin/env python3
"""Lightweight 3D CNN models for video action recognition.

Smaller models that train from scratch on UCF-101.
"""

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """Basic 3D residual block."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleR3D(nn.Module):
    """Simplified 3D ResNet for training from scratch.

    Much smaller than R3D-18 (pretrained).
    """

    def __init__(
        self,
        num_classes=101,
        block=BasicBlock3D,
        layers=[1, 1, 1, 1],  # Number of blocks in each layer
        base_width=32,  # Base channel width
        dropout=0.3,
    ):
        """
        Args:
            num_classes: Number of action classes
            block: Basic block type
            layers: Number of blocks per layer (default [1,1,1,1] = R3D-4)
            base_width: Base channel width (32 = lightweight, 64 = standard)
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = base_width

        # Initial convolution
        self.conv1 = nn.Conv3d(
            3,
            base_width,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # Residual layers
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_width * 8, num_classes)

        self.num_classes = num_classes
        self.layers_config = layers
        self.base_width = base_width

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(channels),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels

        for _ in range(1, blocks):
            layers.append(block(channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) tensor
        Returns:
            (B, num_classes) tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_config(self):
        return {
            "model_name": f"simple_r3d_{sum(self.layers_config)}",
            "num_classes": self.num_classes,
            "layers": self.layers_config,
            "base_width": self.base_width,
            "type": "3dcnn_scratch",
        }


class TinyC3D(nn.Module):
    """
    Tiny C3D - Very lightweight 3D CNN without residual connections.
    Good for quick experiments.
    """

    def __init__(self, num_classes=101, dropout=0.5):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Conv1
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Conv2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Conv3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Conv4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

        # Initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_config(self):
        return {
            "model_name": "tiny_c3d",
            "num_classes": self.num_classes,
            "type": "3dcnn_scratch",
        }


# ============================================================================
# MODEL FACTORY
# ============================================================================


def create_lightweight_model(
    model_name="r3d_4", num_classes=101, dropout=0.3, pretrained=False
):
    """Create lightweight 3D CNN model.

    Args:
        model_name: Model architecture
            - 'r3d_4': 4 layers, 32 base width (~0.8M params)
            - 'r3d_6': 6 layers, 32 base width (~1.2M params)
            - 'r3d_8': 8 layers, 48 base width (~3M params)
            - 'r3d_10': 10 layers, 64 base width (~6M params)
            - 'tiny_c3d': No residual, very small (~0.6M params)
        num_classes: Number of classes
        dropout: Dropout rate
        pretrained: Not used (always train from scratch)

    Returns:
        Model instance
    """
    if model_name == "r3d_4":
        model = SimpleR3D(
            num_classes=num_classes,
            layers=[1, 1, 1, 1],  # 4 blocks
            base_width=32,
            dropout=dropout,
        )

    elif model_name == "r3d_6":
        model = SimpleR3D(
            num_classes=num_classes,
            layers=[1, 1, 2, 2],  # 6 blocks
            base_width=32,
            dropout=dropout,
        )

    elif model_name == "r3d_8":
        model = SimpleR3D(
            num_classes=num_classes,
            layers=[2, 2, 2, 2],  # 8 blocks
            base_width=48,
            dropout=dropout,
        )

    elif model_name == "r3d_10":
        model = SimpleR3D(
            num_classes=num_classes,
            layers=[2, 2, 3, 3],  # 10 blocks
            base_width=64,
            dropout=dropout,
        )

    elif model_name == "tiny_c3d":
        model = TinyC3D(num_classes=num_classes, dropout=dropout)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ============================================================================
# MODEL COMPARISON
# ============================================================================


def compare_models():
    """Compare model sizes and complexity."""

    models = [
        ("tiny_c3d", TinyC3D(num_classes=101)),
        ("r3d_4", SimpleR3D(num_classes=101, layers=[1, 1, 1, 1], base_width=32)),
        ("r3d_6", SimpleR3D(num_classes=101, layers=[1, 1, 2, 2], base_width=32)),
        ("r3d_8", SimpleR3D(num_classes=101, layers=[2, 2, 2, 2], base_width=48)),
        ("r3d_10", SimpleR3D(num_classes=101, layers=[2, 2, 3, 3], base_width=64)),
    ]

    print("\n" + "=" * 80)
    print("Lightweight 3D CNN Model Comparison")
    print("=" * 80)
    print(f"{'Model':<15} {'Parameters':<15} {'Size (MB)':<15} {'Notes'}")
    print("-" * 80)

    for name, model in models:
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / (1024**2)  # Assume float32

        # Test forward pass
        dummy_input = torch.randn(1, 3, 16, 112, 112)
        try:
            _ = model(dummy_input)
            status = "✓ Works"
        except Exception as e:
            status = f"✗ Error: {str(e)[:30]}"

        print(
            f"{name:<15} {params/1e6:>6.2f}M        {size_mb:>6.1f} MB       {status}"
        )

    print("-" * 80)
    print("For reference:")
    print("  R3D-18 (pretrained): 33.2M params (~132 MB)")
    print("  Target for scratch: 1-6M params (better generalization)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    compare_models()
