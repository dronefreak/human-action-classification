"""Training script for action classification with resume support."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from hac.data.dataset import ActionDataset
from hac.models.classifier import create_model
from hac.utils.transforms import get_inference_transforms, get_training_transforms


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
) -> Dict:
    """Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint on

    Returns:
        Dictionary with checkpoint info (epoch, best_val_acc, etc.)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Model weights loaded")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # CRITICAL FIX: Move optimizer states to correct device
        # This fixes "Expected all tensors to be on the same device" error
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"✓ Optimizer state loaded and moved to {device}")

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✓ Scheduler state loaded")

    # Extract training info
    start_epoch = checkpoint.get("epoch", 0)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)

    print(f"✓ Resuming from epoch {start_epoch}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")

    return {
        "start_epoch": start_epoch,
        "best_val_acc": best_val_acc,
        "train_losses": checkpoint.get("train_losses", []),
        "val_losses": checkpoint.get("val_losses", []),
        "val_accuracies": checkpoint.get("val_accuracies", []),
    }


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """Apply mixup augmentation to batch.

    Args:
        x: Input images (batch_size, C, H, W)
        y: Labels (batch_size,)
        alpha: Mixup interpolation strength (0.2-0.4 recommended)
        device: Device for tensors

    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for both images
        lam: Mixing coefficient
    """
    import numpy as np

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    """Training manager with resume support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: str = "cuda",
        output_dir: str = "outputs",
        max_epochs: int = 50,
        start_epoch: int = 0,
        best_val_acc: float = 0.0,
        resume_history: Optional[Dict] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        label_smoothing: float = 0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.start_epoch = start_epoch

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.best_val_acc = best_val_acc

        # Mixup settings
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Metrics tracking - restore from checkpoint if available
        if resume_history:
            self.train_losses = resume_history.get("train_losses", [])
            self.val_losses = resume_history.get("val_losses", [])
            self.val_accuracies = resume_history.get("val_accuracies", [])
        else:
            self.train_losses = []
            self.val_losses = []
            self.val_accuracies = []

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with optional mixup."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply mixup if enabled
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=self.mixup_alpha, device=self.device
                )

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss (with or without mixup)
            if self.use_mixup:
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            # For mixup, approximate accuracy using hard labels
            if self.use_mixup:
                correct += (
                    lam * predicted.eq(labels_a).sum().item()
                    + (1 - lam) * predicted.eq(labels_b).sum().item()
                )
            else:
                correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.max_epochs} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
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
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch + 1}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)

        for epoch in range(self.start_epoch + 1, self.max_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            if self.val_loader:
                val_metrics = self.validate(epoch)
                self.val_losses.append(val_metrics["loss"])
                self.val_accuracies.append(val_metrics["accuracy"])

                print(
                    f"\nEpoch {epoch}: Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )

                # Save best model
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint(epoch, is_best=True)
            else:
                print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

        print("\n" + "=" * 60)
        print("Training complete!")
        if self.val_loader:
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)

        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train action classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_100")
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone weights"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument("--mixup", action="store_true", help="Use mixup augmentation")
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Mixup alpha parameter (0.2-0.4 recommended)",
    )

    args = parser.parse_args()

    # Create datasets
    train_dataset = ActionDataset(
        root_dir=args.data_dir, split="train", transform=get_training_transforms()
    )

    val_dataset = ActionDataset(
        root_dir=args.data_dir, split="val", transform=get_inference_transforms()
    )

    # Create data loaders
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

    # Create model
    model = create_model(
        model_type="action",
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    resume_history = None

    if args.resume:
        checkpoint_info = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
        )
        start_epoch = checkpoint_info["start_epoch"]
        best_val_acc = checkpoint_info["best_val_acc"]
        resume_history = checkpoint_info

        print(f"\n{'='*60}")
        print("Resumed training from checkpoint")
        print(f"{'='*60}\n")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        resume_history=resume_history,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
