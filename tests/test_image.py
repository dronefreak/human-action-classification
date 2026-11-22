"""Image pipeline tests: predictor, trainer, evaluation helpers, metrics (CPU)."""

import cv2
import numpy as np
import pytest
import torch
from conftest import IMAGE_CLASSES
from torch.utils.data import DataLoader

from hac.common.class_names import STANFORD40_CLASSES
from hac.common.metrics import compute_accuracy, compute_metrics
from hac.common.transforms import get_inference_transforms, get_training_transforms
from hac.image.data.dataset import ActionDataset
from hac.image.models.classifier import create_model

ARCH = "mobilenetv3_small_100"  # small timm model, built without downloads


def _save_image_checkpoint(path, num_classes=3, class_names=None):
    model = create_model(model_name=ARCH, num_classes=num_classes, pretrained=False)
    ckpt = {"model_state_dict": model.state_dict(), "config": model.get_config()}
    if class_names is not None:
        ckpt["class_names"] = class_names
    torch.save(ckpt, path)
    return model


def _predictor(path, **kwargs):
    from hac.image.inference.predictor import ActionPredictor

    return ActionPredictor(
        model_path=str(path), device="cpu", use_pose_estimation=False, **kwargs
    )


# ---------------------------------------------------------------------- predictor
def test_predictor_checkpoint_names_and_outputs(tmp_path, rng):
    ckpt = tmp_path / "m.pth"
    _save_image_checkpoint(ckpt, class_names=IMAGE_CLASSES)
    predictor = _predictor(ckpt)
    assert predictor.class_names == IMAGE_CLASSES

    image = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    result = predictor.predict_image(image, top_k=10)
    preds = result["action"]["predictions"]
    assert len(preds) == 3  # top_k clamped to the class count
    assert result["action"]["top_class"] == preds[0]["class"]
    assert sum(p["confidence"] for p in preds) == pytest.approx(1.0, abs=1e-4)

    path = tmp_path / "img.jpg"
    cv2.imwrite(str(path), image)
    assert predictor.predict_image(str(path))["action"]["top_class"] in IMAGE_CLASSES


def test_predictor_defaults_to_sorted_stanford40(tmp_path):
    ckpt = tmp_path / "m.pth"
    _save_image_checkpoint(ckpt, num_classes=40)  # legacy: no stored names
    assert _predictor(ckpt).class_names == STANFORD40_CLASSES


def test_predictor_bad_image_path_message(tmp_path):
    ckpt = tmp_path / "m.pth"
    _save_image_checkpoint(ckpt)
    with pytest.raises(ValueError, match="missing.jpg"):
        _predictor(ckpt).predict_image(str(tmp_path / "missing.jpg"))


def test_predictor_safetensors_needs_architecture_hint(tmp_path):
    from safetensors.torch import save_file

    model = create_model(model_name=ARCH, num_classes=3, pretrained=False)
    weights = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(weights))

    predictor = _predictor(weights, model_name=ARCH, num_classes=3)
    assert predictor.model.model_name == ARCH
    assert predictor.class_names == ["class_0", "class_1", "class_2"]

    # Without the hint the default (resnet34) cannot load these weights
    with pytest.raises(RuntimeError, match="model_name"):
        _predictor(weights)


def test_predictor_video_voting(tmp_path, video_file):
    ckpt = tmp_path / "m.pth"
    _save_image_checkpoint(ckpt, class_names=IMAGE_CLASSES)
    result = _predictor(ckpt).predict_video(str(video_file), sample_rate=4)
    assert result["sampled_frames"] == 3  # frames 0, 4, 8 of 12
    assert result["prediction"] in IMAGE_CLASSES
    assert 0 < result["confidence"] <= 1


# ------------------------------------------------------------------------ trainer
def test_image_trainer_one_epoch_saves_class_names(tmp_path, image_dataset_dir):
    from hac.image.training.train import Trainer

    train = ActionDataset(
        image_dataset_dir, "train", transform=get_training_transforms(64)
    )
    val = ActionDataset(
        image_dataset_dir,
        "test",
        transform=get_inference_transforms(64),
        class_names=train.class_names,
    )
    model = create_model(model_name=ARCH, num_classes=3, pretrained=False)
    trainer = Trainer(
        model=model,
        train_loader=DataLoader(train, batch_size=3),
        val_loader=DataLoader(val, batch_size=3),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        device="cpu",
        output_dir=str(tmp_path / "out"),
        max_epochs=1,
        use_mixup=True,
        class_names=train.class_names,
    )
    trainer.train()

    ckpt = torch.load(tmp_path / "out" / "best.pth", weights_only=False)
    assert ckpt["class_names"] == IMAGE_CLASSES
    assert ckpt["config"]["model_name"] == ARCH
    assert (tmp_path / "out" / "model.safetensors").exists()


# ------------------------------------------------------------------- evaluation
def test_eval_class_resolution(tmp_path, image_dataset_dir):
    from hac.image.inference.evaluate import _resolve_eval_class_names

    split = image_dataset_dir / "test"
    stored = {"class_names": ["z", "y", "x"]}
    assert _resolve_eval_class_names(stored, {"num_classes": 3}, split) == [
        "z",
        "y",
        "x",
    ]
    assert _resolve_eval_class_names({}, {"num_classes": 3}, split) == IMAGE_CLASSES
    # Legacy 40-class checkpoint on a partial split -> canonical Stanford40 list
    assert _resolve_eval_class_names({}, {"num_classes": 40}, split) == (
        STANFORD40_CLASSES
    )
    with pytest.raises(ValueError):
        _resolve_eval_class_names({}, {"num_classes": 7}, split)


def test_per_class_metrics_align_when_classes_absent():
    from hac.image.inference.evaluate import compute_per_class_metrics

    names = ["a", "b", "c"]
    y_true = np.array([0, 0, 2])  # class "b" never appears
    y_pred = np.array([0, 2, 2])
    report = compute_per_class_metrics(y_true, y_pred, names)
    assert report["per_class"]["b"]["support"] == 0
    assert report["per_class"]["c"]["support"] == 1
    assert report["per_class"]["c"]["recall"] == 1.0
    assert report["per_class"]["a"]["recall"] == 0.5


def test_common_metrics():
    preds, targets = [0, 1, 1, 2], [0, 1, 2, 2]
    assert compute_accuracy(preds, targets) == 0.75
    m = compute_metrics(preds, targets)
    assert set(m) == {"accuracy", "precision", "recall", "f1"}
    assert m["accuracy"] == 0.75
