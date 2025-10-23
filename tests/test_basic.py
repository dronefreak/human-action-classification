"""Basic tests to verify package structure."""

import numpy as np
import pytest
import torch


def test_imports():
    """Test that main modules can be imported."""
    from hac import ActionClassifier, ActionPredictor
    from hac.inference.pose_extractor import PoseExtractor
    from hac.models.classifier import create_model

    # Basic sanity checks
    assert callable(create_model), "create_model should be callable"
    assert hasattr(ActionClassifier, "__init__"), "ActionClassifier should be a class"
    assert hasattr(ActionPredictor, "__init__"), "ActionPredictor should be a class"
    assert hasattr(PoseExtractor, "__init__"), "PoseExtractor should be a class"


def test_model_creation():
    """Test model creation."""
    from hac.models.classifier import create_model

    model = create_model(
        model_type="action",
        model_name="mobilenetv3_small_100",
        num_classes=40,
        pretrained=False,
    )

    assert model is not None

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (1, 40)


def test_pose_extractor():
    """Test pose extraction."""
    from hac.inference.pose_extractor import PoseExtractor

    pose_extractor = PoseExtractor()

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # This might return None if no pose detected in random noise
    keypoints = pose_extractor.extract_keypoints(dummy_image)

    # Just verify it doesn't crash
    assert keypoints is None or keypoints.shape == (33, 3)


def test_transforms():
    """Test image transforms."""
    from hac.utils.transforms import get_inference_transforms, get_training_transforms

    train_transform = get_training_transforms()
    infer_transform = get_inference_transforms()

    assert train_transform is not None
    assert infer_transform is not None

    # Test transform
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    transformed = infer_transform(dummy_image)

    assert transformed.shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
