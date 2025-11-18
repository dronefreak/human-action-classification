"""Dataset loader for action classification.

Supports Stanford 40 format and generic image folder structure.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    """Action classification dataset.

    Expected directory structure:
        data_dir/
            train/
                class_1/
                    img1.jpg
                    img2.jpg
                class_2/
                    img1.jpg
            val/
                class_1/
                class_2/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize dataset.

        Args:
            root_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            transform: Image transformations
            class_names: Optional list of class names (will infer if None)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Get split directory
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        # Get class names and create mapping
        if class_names is None:
            self.class_names = sorted(
                [d.name for d in self.split_dir.iterdir() if d.is_dir()]
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load all image paths and labels
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples from {split} split")
        print(f"Number of classes: {len(self.class_names)}")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []

        for class_name in self.class_names:
            class_dir = self.split_dir / class_name

            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]

            # Get all image files
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get item by index.

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.class_names[idx]


class KeypointDataset(Dataset):
    """Dataset that loads pre-extracted pose keypoints.

    Useful for training pose-based action classifiers.
    """

    def __init__(
        self,
        keypoints_file: str,
        labels_file: str,
        transform: Optional[Callable] = None,
    ):
        """Initialize keypoint dataset.

        Args:
            keypoints_file: Path to .npy file with keypoints (N, 33, 2)
            labels_file: Path to .npy file with labels (N,)
            transform: Optional transform for keypoints
        """
        self.keypoints = np.load(keypoints_file)
        self.labels = np.load(labels_file)
        self.transform = transform

        assert len(self.keypoints) == len(
            self.labels
        ), "Keypoints and labels must have same length"

        print(f"Loaded {len(self.keypoints)} keypoint samples")

    def __len__(self) -> int:
        return len(self.keypoints)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        keypoints = self.keypoints[idx]  # (33, 2)
        label = self.labels[idx]

        # Flatten keypoints to (66,) for MLP
        keypoints_flat = keypoints.flatten()

        if self.transform:
            keypoints_flat = self.transform(keypoints_flat)

        return keypoints_flat, label
