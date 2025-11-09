"""Shared metrics for both image and video classification."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_accuracy(predictions, targets):
    """Compute accuracy.

    Args:
        predictions: Array of predicted class indices
        targets: Array of ground truth class indices

    Returns:
        Accuracy as float
    """
    return accuracy_score(targets, predictions)


def compute_metrics(predictions, targets, average="macro"):
    """Compute comprehensive classification metrics.

    Args:
        predictions: Array of predicted class indices
        targets: Array of ground truth class indices
        average: Averaging strategy ('macro', 'weighted', 'micro')

    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    acc = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=average, zero_division=0
    )

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def compute_confusion_matrix(predictions, targets):
    """Compute confusion matrix.

    Args:
        predictions: Array of predicted class indices
        targets: Array of ground truth class indices

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(targets, predictions)
