#!/usr/bin/env python3
"""Train video action classifier on UCF-101."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from hac.common.metrics import compute_metrics
from hac.video.data.augmentations import VideoAugmentation, mixup_criterion
from hac.video.data.dataset import VideoDataset
from hac.video.models.classifier import Video3DCNN

torch.backends.cudnn.benchmark = True  # Auto-tune kernels
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class VideoTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        output_dir,
        max_epochs=50,
        use_augmentation=True,
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs

        # Video augmentation
        if use_augmentation:
            self.video_aug = VideoAugmentation(
                mixup_alpha=mixup_alpha,
                mixup_prob=0.5,
                cutmix_alpha=cutmix_alpha,
                cutmix_prob=0.5,
                frame_drop_rate=0.2,
                frame_drop_prob=0.3,
                temporal_jitter_prob=0.2,
            )
        else:
            self.video_aug = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, epoch):
        """Train for one epoch with video augmentations."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs} [Train]"
        )

        for videos, labels in pbar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Apply video augmentation
            if self.video_aug is not None:
                videos, labels = self.video_aug(videos, labels)

            self.optimizer.zero_grad()

            outputs = self.model(videos)

            # Compute loss (handles both regular and mixed labels)
            loss = mixup_criterion(self.criterion, outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Accuracy (use original labels for mixed samples)
            if isinstance(labels, tuple):
                labels_a, labels_b, lam = labels
                _, predicted = outputs.max(1)
                total += labels_a.size(0)
                correct += (
                    lam * predicted.eq(labels_a).sum().item()
                    + (1 - lam) * predicted.eq(labels_b).sum().item()
                )
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            acc = 100.0 * correct / total
            pbar.set_postfix(
                {"loss": running_loss / (pbar.n + 1), "acc": f"{acc:.2f}%"}
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, desc=f"Epoch {epoch+1}/{self.max_epochs} [Val]"
            )

            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                acc = 100.0 * correct / total
                pbar.set_postfix(
                    {"loss": running_loss / (pbar.n + 1), "acc": f"{acc:.2f}%"}
                )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        # Compute detailed metrics
        metrics = compute_metrics(all_predictions, all_targets)

        return epoch_loss, epoch_acc, metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "metrics": metrics,
            "config": (
                self.model.get_config() if hasattr(self.model, "get_config") else {}
            ),
        }

        # Save latest
        latest_path = self.output_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (accuracy: {self.best_val_acc:.2f}%)")

    def train(self):
        """Run full training loop."""
        print(f"\nStarting training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}\n")

        for epoch in range(self.max_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Print summary
            print(f"\nEpoch {epoch+1}/{self.max_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(
                f"  Val F1: {metrics['f1']:.4f}, Val Precision: {metrics['precision']:.4f}"  # noqa: E501
            )

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_checkpoint(epoch, metrics, is_best)

        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_acc": self.best_val_acc,
        }

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved to: {self.output_dir / 'best.pth'}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train video action classifier")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="UCF-101 directory")
    parser.add_argument(
        "--train_split", type=str, default="train", help="Train split folder name"
    )
    parser.add_argument(
        "--val_split", type=str, default="test", help="Val split folder name"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="r3d_18",
        choices=["r3d_18", "mc3_18", "r2plus1d_18"],
        help="3D CNN model",
    )
    parser.add_argument(
        "--num_classes", type=int, default=101, help="Number of classes"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )

    # Video sampling
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument(
        "--frame_interval", type=int, default=2, help="Frame sampling interval"
    )
    parser.add_argument("--spatial_size", type=int, default=112, help="Spatial size")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--step_size", type=int, default=20, help="LR scheduler step size"
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Use video augmentations (MixUp, CutMix, etc.)",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_false",
        dest="use_augmentation",
        help="Disable video augmentations",
    )
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.4, help="MixUp alpha parameter"
    )
    parser.add_argument(
        "--cutmix_alpha", type=float, default=1.0, help="CutMix alpha parameter"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=4,
        help="Label smoothing value for the cross-entropy loss",
    )
    # System
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/video", help="Output directory"
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 171)),
            transforms.CenterCrop(args.spatial_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
        ]
    )

    # Datasets
    data_dir = Path(args.data_dir)

    # Auto-detect mode: check if we have frame directories or video files
    train_dir = data_dir / args.train_split
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    # Check first class directory to determine mode
    first_class = next(train_dir.iterdir())
    first_item = next(first_class.iterdir())
    mode = "frames" if first_item.is_dir() else "videos"

    print(f"Detected mode: {mode}")

    train_dataset = VideoDataset(
        root_dir=train_dir,
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
        transform=train_transform,
        mode=mode,
    )

    val_dataset = VideoDataset(
        root_dir=data_dir / args.val_split,
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
        transform=val_transform,
        mode=mode,
    )

    print("\nDataset:")
    print(f"  Train: {len(train_dataset)} videos")
    print(f"  Val: {len(val_dataset)} videos")
    print(f"  Classes: {len(train_dataset.classes)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = Video3DCNN(
        num_classes=args.num_classes, model_name=args.model, pretrained=args.pretrained
    )
    model = model.to(device)

    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # Trainer
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        use_augmentation=args.use_augmentation,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
