"""Human Action Classification - Image and Video."""

__version__ = "2.0.0"

# Common utilities
from .common.metrics import compute_accuracy, compute_metrics

# Image module (backward compatibility)
from .image.inference.predictor import ActionPredictor as ImagePredictor
from .image.models.classifier import ActionClassifier

# Video module
from .video.models.classifier import Video3DCNN

__all__ = [
    "ImagePredictor",
    "ActionClassifier",
    "Video3DCNN",
    "compute_accuracy",
    "compute_metrics",
]
