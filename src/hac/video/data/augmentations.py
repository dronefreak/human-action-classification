#!/usr/bin/env python3
"""Video-specific augmentation techniques for action recognition.

Includes: VideoMixUp, VideoCutMix, FrameDrop, TemporalJitter, etc.
"""

import random

import numpy as np
import torch


class VideoMixUp:
    """MixUp augmentation for video data. Mixes two videos and their labels.

    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    """

    def __init__(self, alpha=0.4, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, videos, labels):
        """Apply mixup to batch of videos.

        Args:
            videos: (B, C, T, H, W) tensor
            labels: (B,) tensor

        Returns:
            Mixed videos, mixed labels (as soft labels)
        """
        if random.random() > self.prob:
            return videos, labels

        batch_size = videos.size(0)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # Random permutation
        index = torch.randperm(batch_size).to(videos.device)

        # Mix videos
        mixed_videos = lam * videos + (1 - lam) * videos[index]

        # Mix labels (soft labels)
        labels_a = labels
        labels_b = labels[index]

        return mixed_videos, (labels_a, labels_b, lam)


class VideoCutMix:
    """CutMix augmentation for video data. Cuts and pastes spatiotemporal regions
    between videos.

    Reference: CutMix: Regularization Strategy to
    Train Strong Classifiers (Yun et al., 2019)
    """

    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, videos, labels):
        """Apply cutmix to batch of videos.

        Args:
            videos: (B, C, T, H, W) tensor
            labels: (B,) tensor

        Returns:
            Mixed videos, mixed labels
        """
        if random.random() > self.prob:
            return videos, labels

        batch_size, C, T, H, W = videos.size()

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # Random permutation
        index = torch.randperm(batch_size).to(videos.device)

        # Random bounding box (spatial + temporal)
        cut_ratio = np.sqrt(1.0 - lam)

        # Spatial cut
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Temporal cut
        cut_t = int(T * cut_ratio)
        ct = np.random.randint(T)
        t1 = np.clip(ct - cut_t // 2, 0, T)
        t2 = np.clip(ct + cut_t // 2, 0, T)

        # Mix videos
        mixed_videos = videos.clone()
        mixed_videos[:, :, t1:t2, y1:y2, x1:x2] = videos[index, :, t1:t2, y1:y2, x1:x2]

        # Adjust lambda based on actual cut size
        lam = 1 - ((x2 - x1) * (y2 - y1) * (t2 - t1) / (W * H * T))

        # Mix labels
        labels_a = labels
        labels_b = labels[index]

        return mixed_videos, (labels_a, labels_b, lam)


class FrameDrop:
    """Randomly drop frames from video to improve temporal robustness.

    Forces model to learn from incomplete temporal information.
    """

    def __init__(self, drop_rate=0.2, prob=0.5):
        """
        Args:
            drop_rate: Fraction of frames to drop (0-1)
            prob: Probability of applying frame drop
        """
        self.drop_rate = drop_rate
        self.prob = prob

    def __call__(self, videos):
        """Randomly drop frames from videos.

        Args:
            videos: (B, C, T, H, W) tensor

        Returns:
            Videos with dropped frames (replaced with black or previous frame)
        """
        if random.random() > self.prob:
            return videos

        batch_size, C, T, H, W = videos.size()

        # Create mask for frames to keep
        keep_mask = torch.rand(batch_size, T) > self.drop_rate
        keep_mask = keep_mask.to(videos.device)

        # Apply mask
        for b in range(batch_size):
            for t in range(T):
                if not keep_mask[b, t]:
                    if t > 0:
                        # Replace with previous frame
                        videos[b, :, t] = videos[b, :, t - 1]
                    else:
                        # Replace with zeros (first frame)
                        videos[b, :, t] = 0

        return videos


class TemporalJitter:
    """Randomly shuffle or permute frame order to improve temporal invariance."""

    def __init__(self, max_shift=2, prob=0.3):
        """
        Args:
            max_shift: Maximum frames to shift
            prob: Probability of applying jitter
        """
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, videos):
        """Apply temporal jitter to videos.

        Args:
            videos: (B, C, T, H, W) tensor

        Returns:
            Temporally jittered videos
        """
        if random.random() > self.prob:
            return videos

        batch_size, C, T, H, W = videos.size()

        for b in range(batch_size):
            # Random temporal shift
            shift = random.randint(-self.max_shift, self.max_shift)
            videos[b] = torch.roll(videos[b], shift, dims=1)

        return videos


class VideoAugmentation:
    """Combined video augmentation pipeline.

    Applies multiple video-specific augmentations.
    """

    def __init__(
        self,
        mixup_alpha=0.4,
        mixup_prob=0.5,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
        frame_drop_rate=0.2,
        frame_drop_prob=0.3,
        temporal_jitter_prob=0.2,
    ):
        """Initialize video augmentation pipeline.

        Args:
            mixup_alpha: MixUp alpha parameter
            mixup_prob: Probability of applying MixUp
            cutmix_alpha: CutMix alpha parameter
            cutmix_prob: Probability of applying CutMix
            frame_drop_rate: Rate of frames to drop
            frame_drop_prob: Probability of applying FrameDrop
            temporal_jitter_prob: Probability of applying temporal jitter
        """
        self.mixup = VideoMixUp(alpha=mixup_alpha, prob=mixup_prob)
        self.cutmix = VideoCutMix(alpha=cutmix_alpha, prob=cutmix_prob)
        self.frame_drop = FrameDrop(drop_rate=frame_drop_rate, prob=frame_drop_prob)
        self.temporal_jitter = TemporalJitter(prob=temporal_jitter_prob)

        # Use only one of mixup or cutmix per batch
        self.use_mixup_or_cutmix = True

    def __call__(self, videos, labels):
        """Apply video augmentations.

        Args:
            videos: (B, C, T, H, W) tensor
            labels: (B,) tensor

        Returns:
            Augmented videos, augmented labels
        """
        # Apply frame-level augmentations (no label mixing)
        videos = self.frame_drop(videos)
        videos = self.temporal_jitter(videos)

        # Apply mixing augmentation (label mixing)
        if self.use_mixup_or_cutmix:
            if random.random() < 0.5:
                videos, labels = self.mixup(videos, labels)
            else:
                videos, labels = self.cutmix(videos, labels)

        return videos, labels


def mixup_criterion(criterion, pred, labels_tuple):
    """Compute loss for mixup/cutmix.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions (B, num_classes)
        labels_tuple: Either regular labels (B,)
        or mixed labels (labels_a, labels_b, lam)

    Returns:
        Loss value
    """
    if isinstance(labels_tuple, tuple):
        # Mixed labels
        labels_a, labels_b, lam = labels_tuple
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
    else:
        # Regular labels
        return criterion(pred, labels_tuple)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def example_usage():
    """Example of how to use video augmentations in training."""
    from torch.optim import Adam

    # Create augmentation pipeline
    video_aug = VideoAugmentation(
        mixup_alpha=0.4,
        mixup_prob=0.5,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
        frame_drop_rate=0.2,
        frame_drop_prob=0.3,
    )

    # Dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 64, kernel_size=3),
        torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 101),
    )

    optimizer = Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Dummy batch
    videos = torch.randn(8, 3, 16, 112, 112)  # (B, C, T, H, W)
    labels = torch.randint(0, 101, (8,))

    # Training step with augmentation
    model.train()

    # Apply video augmentation
    videos_aug, labels_aug = video_aug(videos, labels)

    # Forward pass
    outputs = model(videos_aug)

    # Compute loss (handles both regular and mixed labels)
    loss = mixup_criterion(criterion, outputs, labels_aug)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    example_usage()
