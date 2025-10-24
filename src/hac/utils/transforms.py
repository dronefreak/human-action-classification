"""Image transformation utilities."""

import numpy as np
from torchvision import transforms


def get_training_transforms(input_size: int = 224):
    """Get training transforms with augmentation.

    Args:
        input_size: Image size (assumes square)

    Returns:
        torchvision transforms
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_inference_transforms(input_size: int = 224):
    """Get inference transforms (no augmentation).

    Args:
        input_size: Image size (assumes square)

    Returns:
        torchvision transforms
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)

    return tensor
