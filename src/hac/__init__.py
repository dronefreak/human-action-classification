"""Human Action Classification v2.0.

Modern action recognition using MediaPipe pose estimation and PyTorch. Completely
rewritten from the TensorFlow 1.13 version.
"""

__version__ = "2.0.0"
__author__ = "Saumya Kumaar Saksena"

from hac.inference.predictor import ActionPredictor
from hac.models.classifier import ActionClassifier

__all__ = ["ActionPredictor", "ActionClassifier"]
