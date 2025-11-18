"""Evaluation script for computing comprehensive metrics on trained models.

Run this AFTER training to get detailed performance analysis.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from hac.common.transforms import get_inference_transforms
from hac.image.data.dataset import ActionDataset
from hac.image.models.classifier import create_model


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint and extract config."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint.get("config", {})
    if not config:
        # Fallback to defaults if no config
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            "model_name": "resnet34",  # You'll need to specify this
            "num_classes": 40,
        }

    # Create model
    model = create_model(
        model_type="action",
        model_name=config.get("model_name", "resnet34"),
        num_classes=config.get("num_classes", 40),
        pretrained=False,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded: {config.get('model_name', 'unknown')}")

    return model, config, checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: str = "cuda",
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate model and collect predictions.

    Returns:
        metrics: Dict with accuracy
        all_preds: Array of predictions
        all_labels: Array of ground truth labels
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("\nRunning inference...")
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        probs = torch.softmax(outputs, dim=-1)
        _, predicted = outputs.max(1)

        # Collect predictions
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute accuracy
    accuracy = (all_preds == all_labels).mean() * 100

    metrics = {
        "accuracy": float(accuracy),
        "num_samples": len(all_labels),
    }

    print(f"✓ Accuracy: {accuracy:.2f}%")

    return metrics, all_preds, all_labels, all_probs


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict:
    """Compute precision, recall, F1 for each class."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred) * 100

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }

    summary = {
        "overall_accuracy": float(overall_accuracy),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1),
        },
        "per_class": per_class_metrics,
    }

    print(f"\n✓ Macro F1-Score: {macro_f1:.4f}")
    print(f"✓ Weighted F1-Score: {weighted_f1:.4f}")

    return summary


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> np.ndarray:
    """Compute confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    print(f"✓ Confusion matrix computed: {cm.shape}")

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = False,
    figsize: Tuple[int, int] = (20, 18),
):
    """Plot and save confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count" if not normalize else "Proportion"},
    )

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Confusion matrix plot saved: {output_path}")
    plt.close()


def plot_per_class_metrics(metrics: Dict, output_path: Path, top_k: int = 10):
    """Plot top-k and bottom-k classes by F1 score."""
    per_class = metrics["per_class"]

    # Sort by F1 score
    sorted_classes = sorted(
        per_class.items(), key=lambda x: x[1]["f1_score"], reverse=True
    )

    # Get top and bottom k
    top_classes = sorted_classes[:top_k]
    bottom_classes = sorted_classes[-top_k:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Top K classes
    names = [x[0] for x in top_classes]
    f1_scores = [x[1]["f1_score"] for x in top_classes]

    ax1.barh(names, f1_scores, color="green", alpha=0.7)
    ax1.set_xlabel("F1 Score")
    ax1.set_title(f"Top {top_k} Classes by F1 Score")
    ax1.set_xlim([0, 1])

    # Bottom K classes
    names = [x[0] for x in bottom_classes]
    f1_scores = [x[1]["f1_score"] for x in bottom_classes]

    ax2.barh(names, f1_scores, color="red", alpha=0.7)
    ax2.set_xlabel("F1 Score")
    ax2.set_title(f"Bottom {top_k} Classes by F1 Score")
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Per-class metrics plot saved: {output_path}")
    plt.close()


def find_worst_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
) -> List[Dict]:
    """Find the most confident wrong predictions."""
    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        return []

    # Get confidence of wrong predictions
    wrong_confidences = y_probs[wrong_indices, y_pred[wrong_indices]]

    # Sort by confidence (most confident mistakes)
    sorted_indices = np.argsort(wrong_confidences)[::-1][:top_k]

    worst_predictions = []
    for idx in sorted_indices:
        sample_idx = wrong_indices[idx]
        worst_predictions.append(
            {
                "sample_index": int(sample_idx),
                "true_label": class_names[y_true[sample_idx]],
                "predicted_label": class_names[y_pred[sample_idx]],
                "confidence": float(wrong_confidences[idx]),
            }
        )

    return worst_predictions


def save_metrics(
    metrics: Dict, output_dir: Path, formats: List[str] = ["json", "yaml", "txt"]
):
    """Save metrics in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    metrics["evaluated_at"] = datetime.now().isoformat()

    # Save JSON
    if "json" in formats:
        json_path = output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved: {json_path}")

    # Save YAML
    if "yaml" in formats:
        yaml_path = output_dir / "metrics.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Metrics saved: {yaml_path}")

    # Save human-readable TXT
    if "txt" in formats:
        txt_path = output_dir / "metrics.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            # Basic info
            f.write(f"Evaluated at: {metrics.get('evaluated_at', 'N/A')}\n")
            f.write(f"Checkpoint: {metrics.get('checkpoint_path', 'N/A')}\n")
            f.write(f"Model: {metrics.get('model_name', 'N/A')}\n")
            f.write(f"Dataset split: {metrics.get('split', 'N/A')}\n")
            f.write(f"Number of samples: {metrics.get('num_samples', 'N/A')}\n")
            f.write("\n")

            # Overall metrics
            f.write("-" * 60 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy: {metrics.get('accuracy', 0):.2f}%\n")

            if "classification_report" in metrics:
                report = metrics["classification_report"]
                f.write("\nMacro Average:\n")
                f.write(f"  Precision: {report['macro_avg']['precision']:.4f}\n")
                f.write(f"  Recall:    {report['macro_avg']['recall']:.4f}\n")
                f.write(f"  F1-Score:  {report['macro_avg']['f1_score']:.4f}\n")

                f.write("\nWeighted Average:\n")
                f.write(f"  Precision: {report['weighted_avg']['precision']:.4f}\n")
                f.write(f"  Recall:    {report['weighted_avg']['recall']:.4f}\n")
                f.write(f"  F1-Score:  {report['weighted_avg']['f1_score']:.4f}\n")

            # Per-class metrics (top 5 and bottom 5)
            if "classification_report" in metrics:
                per_class = metrics["classification_report"]["per_class"]
                sorted_classes = sorted(
                    per_class.items(), key=lambda x: x[1]["f1_score"], reverse=True
                )

                f.write("\n" + "-" * 60 + "\n")
                f.write("TOP 5 BEST PERFORMING CLASSES\n")
                f.write("-" * 60 + "\n")
                for i, (class_name, scores) in enumerate(sorted_classes[:5], 1):
                    f.write(f"{i}. {class_name}\n")
                    f.write(f"   Precision: {scores['precision']:.4f} | ")
                    f.write(f"Recall: {scores['recall']:.4f} | ")
                    f.write(f"F1: {scores['f1_score']:.4f}\n")

                f.write("\n" + "-" * 60 + "\n")
                f.write("BOTTOM 5 WORST PERFORMING CLASSES\n")
                f.write("-" * 60 + "\n")
                for i, (class_name, scores) in enumerate(sorted_classes[-5:], 1):
                    f.write(f"{i}. {class_name}\n")
                    f.write(f"   Precision: {scores['precision']:.4f} | ")
                    f.write(f"Recall: {scores['recall']:.4f} | ")
                    f.write(f"F1: {scores['f1_score']:.4f}\n")

            # Worst predictions
            if "worst_predictions" in metrics:
                f.write("\n" + "-" * 60 + "\n")
                f.write("TOP 10 MOST CONFIDENT WRONG PREDICTIONS\n")
                f.write("-" * 60 + "\n")
                for i, pred in enumerate(metrics["worst_predictions"], 1):
                    f.write(f"{i}. True: {pred['true_label']} | ")
                    f.write(f"Predicted: {pred['predicted_label']} | ")
                    f.write(f"Confidence: {pred['confidence']:.4f}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"✓ Metrics saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model and compute comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set
  python evaluate.py \\
      --checkpoint outputs/resnet34_best.pth \\
      --data_dir stanford40/ \\
      --split test \\
      --output_dir evaluation_results/resnet34

  # Evaluate with specific model name (if not in checkpoint)
  python evaluate.py \\
      --checkpoint best.pth \\
      --data_dir stanford40/ \\
      --model_name resnet34 \\
      --num_classes 40
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate (test/val)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Optional overrides if checkpoint doesn't have config
    parser.add_argument(
        "--model_name", type=str, help="Model architecture (if not in checkpoint)"
    )
    parser.add_argument(
        "--num_classes", type=int, help="Number of classes (if not in checkpoint)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load model
    model, config, checkpoint = load_model_from_checkpoint(args.checkpoint, args.device)

    # Override config if specified
    if args.model_name:
        config["model_name"] = args.model_name
    if args.num_classes:
        config["num_classes"] = args.num_classes

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = ActionDataset(
        root_dir=args.data_dir, split=args.split, transform=get_inference_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = dataset.class_names
    print(f"✓ Loaded {len(dataset)} samples, {len(class_names)} classes")

    # Evaluate
    basic_metrics, y_pred, y_true, y_probs = evaluate_model(
        model, dataloader, class_names, args.device
    )

    # Compute detailed metrics
    print("\nComputing detailed metrics...")
    classification_report = compute_per_class_metrics(y_true, y_pred, class_names)

    # Compute confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, class_names)

    # Find worst predictions
    worst_preds = find_worst_predictions(y_true, y_pred, y_probs, class_names, top_k=10)

    # Compile all metrics
    all_metrics = {
        "checkpoint_path": args.checkpoint,
        "model_name": config.get("model_name", "unknown"),
        "num_classes": config.get("num_classes", len(class_names)),
        "split": args.split,
        "num_samples": len(dataset),
        "accuracy": basic_metrics["accuracy"],
        "classification_report": classification_report,
        "worst_predictions": worst_preds,
        "training_epochs": checkpoint.get("epoch", "N/A"),
        "best_val_acc": checkpoint.get("best_val_acc", "N/A"),
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics in multiple formats
    print("\nSaving results...")
    save_metrics(all_metrics, output_dir, formats=["json", "yaml", "txt"])

    # Save confusion matrix as numpy array
    np.save(output_dir / "confusion_matrix.npy", cm)
    print(f"✓ Confusion matrix saved: {output_dir / 'confusion_matrix.npy'}")

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names, output_dir / "confusion_matrix.png", normalize=False
    )
    plot_confusion_matrix(
        cm, class_names, output_dir / "confusion_matrix_normalized.png", normalize=True
    )

    # Plot per-class metrics
    plot_per_class_metrics(classification_report, output_dir / "per_class_metrics.png")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey Metrics:")
    print(f"  Accuracy: {all_metrics['accuracy']:.2f}%")
    print(f"  Macro F1: {classification_report['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1: {classification_report['weighted_avg']['f1_score']:.4f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
