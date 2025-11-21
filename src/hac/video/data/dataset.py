#!/usr/bin/env python3
"""
Video dataset for UCF-101 - handles both raw videos and extracted frames.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Dataset for video action recognition (UCF-101).

    Supports both:
    - Raw video files (.avi, .mp4)
    - Extracted frame directories
    """

    def __init__(
        self,
        root_dir,
        num_frames=16,
        frame_interval=2,
        transform=None,
        mode="frames",  # 'frames' or 'videos'
    ):
        """Initialize video dataset.

        Args:
            root_dir: Root directory with class subdirectories
            num_frames: Number of frames to sample per clip
            frame_interval: Stride between sampled frames (ignored for frames mode)
            transform: Transform to apply to each frame
            mode: 'frames' (directories) or 'videos' (.avi files)
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.transform = transform
        self.mode = mode

        # Get all samples
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.is_training = "train" in str(self.root_dir).lower()

        # Scan for videos or frame directories
        self._load_samples()

        if len(self.samples) == 0:
            print(f"\n⚠️  WARNING: No samples found in {root_dir}")
            print(f"   Looking for: {self.mode}")
            print(f"   Classes found: {len(self.classes)}")
            if len(self.classes) > 0:
                sample_class = self.root_dir / self.classes[0]
                print(f"   Sample class dir: {sample_class}")
                print(f"   Contents: {list(sample_class.iterdir())[:5]}")

    def _load_samples(self):
        """Load all video samples."""
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_idx = self.class_to_idx[class_dir.name]

            if self.mode == "videos":
                # Look for video files
                for ext in [".avi", ".mp4", ".mkv"]:
                    for video_path in class_dir.glob(f"*{ext}"):
                        self.samples.append((video_path, class_idx))
            else:
                # Look for frame directories
                for item in class_dir.iterdir():
                    if item.is_dir():
                        # Check if it contains frame images
                        frames = list(item.glob("frame_*.jpg")) + list(
                            item.glob("*.jpg")
                        )
                        if len(frames) >= self.num_frames:
                            self.samples.append((item, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self.mode == "videos":
            # Load from video file
            frames = self._load_video(path)
        else:
            # Load from frame directory
            frames = self._load_frames_from_dir(path)

        # Sample frames
        frames = self._sample_frames(frames, is_training=self.is_training)

        # Apply transforms
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # if isinstance(frame, np.ndarray):
                # frame = Image.fromarray(frame)
                frame = self.transform(frame)
                transformed_frames.append(frame)
            frames = transformed_frames

        # Stack: (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)

        return video_tensor, label

    def _load_video(self, video_path):
        """Load frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        return frames

    def _load_frames_from_dir(self, frame_dir):
        """Load frames from directory."""
        # Get all frame images
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))

        if len(frame_paths) == 0:
            # Try without frame_ prefix
            frame_paths = sorted(frame_dir.glob("*.jpg"))

        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frame_dir}")

        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        return frames

    def _sample_frames(self, frames, is_training=False):
        """Sample num_frames with temporal randomness during training."""
        total_frames = len(frames)

        if total_frames <= self.num_frames:
            # Pad short videos by repeating
            indices = np.arange(total_frames)
            indices = np.tile(indices, int(np.ceil(self.num_frames / total_frames)))
            indices = indices[: self.num_frames]
        else:
            if is_training:
                # Random start + random stride (with frame_interval)
                max_offset = total_frames - self.num_frames * self.frame_interval
                if max_offset <= 0:
                    # Fall back to uniform if video too short
                    indices = np.linspace(0, total_frames - 1, self.num_frames).astype(
                        int
                    )
                else:
                    start = np.random.randint(0, max_offset + 1)
                    indices = start + self.frame_interval * np.arange(self.num_frames)
            else:
                # Center sampling for validation
                start = (total_frames - self.num_frames * self.frame_interval) // 2
                if start < 0:
                    start = 0
                indices = start + self.frame_interval * np.arange(self.num_frames)

        # Clip indices to valid range
        indices = np.clip(indices, 0, total_frames - 1)
        return [frames[i] for i in indices]


def test_dataset():
    """Test dataset loading."""
    import argparse

    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="frames", choices=["frames", "videos"]
    )
    args = parser.parse_args()

    # Simple transform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ]
    )

    # Create dataset
    dataset = VideoDataset(
        root_dir=args.data_dir, num_frames=16, transform=transform, mode=args.mode
    )

    print("\nDataset loaded:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Classes: {len(dataset.classes)}")
    print(f"  Mode: {args.mode}")

    if len(dataset) > 0:
        # Test loading one sample
        video, label = dataset[0]
        print(f"\nSample shape: {video.shape}")
        print("Expected: (C=3, T=16, H=112, W=112)")
        print(f"Label: {label} ({dataset.classes[label]})")


if __name__ == "__main__":
    test_dataset()
