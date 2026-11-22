"""CLI entry points and MediaPipe pose-extractor lifecycle."""

import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from hac import cli
from hac.image.inference.pose_extractor import PoseClassifier, PoseExtractor


def test_hac_train_points_at_real_modules(capsys):
    cli.train()
    out = capsys.readouterr().out
    assert "hac.image.training.train" in out
    assert "hac.video.training.train" in out


def test_infer_requires_an_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hac-infer", "--model", "x.pth"])
    with pytest.raises(SystemExit):
        cli.infer()


def test_infer_image_end_to_end(tmp_path, monkeypatch, capsys, rng):
    from hac.image.models.classifier import create_model

    model = create_model(
        model_name="mobilenetv3_small_100", num_classes=3, pretrained=False
    )
    ckpt = tmp_path / "m.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.get_config(),
            "class_names": ["a", "b", "c"],
        },
        ckpt,
    )
    image = tmp_path / "x.jpg"
    cv2.imwrite(str(image), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hac-infer",
            "--image",
            str(image),
            "--model",
            str(ckpt),
            "--device",
            "cpu",
            "--top_k",
            "2",
        ],
    )
    cli.infer()
    out = capsys.readouterr().out
    assert "Top 2 Action Predictions" in out
    assert out.count(": 0.") + out.count(": 1.") >= 2


def test_video_training_cli_choices_match_registry():
    from hac.video.models.classifier import MODEL_REGISTRY

    result = subprocess.run(
        [sys.executable, "-m", "hac.video.training.train", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    for name in MODEL_REGISTRY:
        assert name in result.stdout


# ------------------------------------------------------------------ pose extractor
def test_pose_extractor_close_is_idempotent_and_context_managed():
    with PoseExtractor(static_image_mode=True) as extractor:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        assert extractor.extract_keypoints(blank) is None
    assert extractor.pose is None
    extractor.close()  # second close must be a no-op


def test_uncaught_error_with_live_extractor_exits_promptly():
    # Regression: MediaPipe's close() used to deadlock in __del__ at interpreter
    # shutdown, so any uncaught error hung the process (e.g. a failing hac-infer).
    code = (
        "from hac.image.inference.pose_extractor import PoseExtractor\n"
        "p = PoseExtractor(static_image_mode=True)\n"
        "raise RuntimeError('boom')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 1
    assert "boom" in result.stderr


def _keypoints(points):
    kp = np.zeros((33, 3))
    kp[:, 2] = 1.0
    for idx, (x, y) in points.items():
        kp[idx, :2] = (x, y)
    return kp


def test_rule_based_pose_classifier():
    # Indices: shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28 (y grows down)
    standing = _keypoints(
        {11: (0.45, 0.2), 12: (0.55, 0.2), 23: (0.45, 0.5), 24: (0.55, 0.5),
         25: (0.45, 0.7), 26: (0.55, 0.7), 27: (0.45, 0.9), 28: (0.55, 0.9)}
    )  # fmt: skip
    lying = _keypoints(
        {11: (0.1, 0.5), 12: (0.1, 0.55), 23: (0.5, 0.5), 24: (0.5, 0.55),
         25: (0.7, 0.5), 26: (0.7, 0.55), 27: (0.9, 0.5), 28: (0.9, 0.55)}
    )  # fmt: skip
    sitting = _keypoints(
        {11: (0.45, 0.2), 12: (0.55, 0.2), 23: (0.45, 0.5), 24: (0.55, 0.5),
         25: (0.45, 0.65), 26: (0.55, 0.65), 27: (0.45, 0.7), 28: (0.55, 0.7)}
    )  # fmt: skip
    assert PoseClassifier.classify_pose(standing) == "standing"
    assert PoseClassifier.classify_pose(lying) == "lying"
    assert PoseClassifier.classify_pose(sitting) == "sitting"
