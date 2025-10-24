"""Training script for action classification."""

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


class Trainer:
    """Training manager."""

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

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        _ = 100.0 * correct / total

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
            "best_val_acc": self.best_val_acc,
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
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)

        for epoch in range(1, self.max_epochs + 1):
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
    parser.add_argument("--freeze_backbone", type=bool, default=False)

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
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
