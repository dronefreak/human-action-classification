"""Video pipeline tests: model, augmentations, dataset, predictor, trainer (CPU)."""

import pytest
import torch
from conftest import VIDEO_CLASSES
from torch.utils.data import DataLoader
from torchvision import transforms

from hac.video.data.augmentations import (
    FrameDrop,
    TemporalJitter,
    VideoAugmentation,
    VideoCutMix,
    VideoMixUp,
    mixup_criterion,
)
from hac.video.data.dataset import VideoDataset
from hac.video.models.classifier import MODEL_REGISTRY, Video3DCNN

CLIP = (2, 3, 4, 32, 32)  # (B, C, T, H, W) small enough for fast CPU tests


# --------------------------------------------------------------------------- model
@pytest.mark.parametrize("name", ["r3d_18", "mc3_18", "r2plus1d_18"])
def test_resnet3d_forward_and_config(name):
    model = Video3DCNN(num_classes=7, model_name=name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(*CLIP))
    assert out.shape == (2, 7)
    assert model.get_config()["model_name"] == name
    assert model.get_config()["num_classes"] == 7


def test_transformer_head_replaced():
    model = Video3DCNN(num_classes=5, model_name="mvit_v2_s", pretrained=False)
    assert isinstance(model.backbone.head, torch.nn.Linear)
    assert model.backbone.head.out_features == 5


def test_freeze_backbone_leaves_only_classifier_trainable():
    model = Video3DCNN(num_classes=4, model_name="r3d_18", pretrained=False)
    model.freeze_backbone()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"backbone.fc.weight", "backbone.fc.bias"}


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        Video3DCNN(model_name="mvit_v1_s", pretrained=False)
    assert "mvit_v2_s" in MODEL_REGISTRY and "mvit_v1_s" not in MODEL_REGISTRY


# ------------------------------------------------------------------- augmentations
def test_mixup_and_cutmix_produce_valid_mixed_labels():
    videos = torch.randn(*CLIP)
    labels = torch.tensor([0, 1])
    for aug in (VideoMixUp(prob=1.0), VideoCutMix(prob=1.0)):
        mixed, (la, lb, lam) = aug(videos, labels)
        assert mixed.shape == videos.shape
        assert 0.0 <= lam <= 1.0
        assert la.shape == lb.shape == labels.shape


def test_augmentations_disabled_with_zero_prob():
    videos = torch.randn(*CLIP)
    labels = torch.tensor([0, 1])
    out, out_labels = VideoMixUp(prob=0.0)(videos, labels)
    assert out is videos and out_labels is labels
    assert FrameDrop(prob=0.0)(videos) is videos
    assert TemporalJitter(prob=0.0)(videos) is videos


def test_frame_drop_and_jitter_keep_shape():
    videos = torch.randn(*CLIP)
    assert FrameDrop(drop_rate=0.5, prob=1.0)(videos.clone()).shape == videos.shape
    assert TemporalJitter(prob=1.0)(videos.clone()).shape == videos.shape


def test_mixup_criterion_handles_plain_and_mixed_labels():
    criterion = torch.nn.CrossEntropyLoss()
    logits = torch.randn(2, 3)
    labels = torch.tensor([0, 2])
    plain = mixup_criterion(criterion, logits, labels)
    assert torch.isclose(plain, criterion(logits, labels))
    # lam=1 must reduce to the loss on labels_a
    mixed = mixup_criterion(criterion, logits, (labels, labels.flip(0), 1.0))
    assert torch.isclose(mixed, plain)
    videos, out = VideoAugmentation()(torch.randn(*CLIP), labels)
    assert videos.shape == CLIP


# ------------------------------------------------------------------------ dataset
def _frame_transform():
    return transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor()]
    )


def test_video_dataset_frames_mode(frames_dataset_dir):
    ds = VideoDataset(
        frames_dataset_dir / "train", num_frames=4, transform=_frame_transform()
    )
    assert ds.classes == VIDEO_CLASSES
    clip, label = ds[0]
    assert clip.shape == (3, 4, 32, 32)
    assert 0 <= label < len(VIDEO_CLASSES)


def test_video_dataset_videos_mode_repeats_short_clips(tmp_path, video_file):
    cls_dir = tmp_path / "vids" / "Archery"
    cls_dir.mkdir(parents=True)
    video_file.rename(cls_dir / "a.avi")
    ds = VideoDataset(
        tmp_path / "vids", num_frames=16, transform=_frame_transform(), mode="videos"
    )
    clip, _ = ds[0]
    assert clip.shape == (3, 16, 32, 32)  # 12-frame clip tiled up to 16


# ---------------------------------------------------------------------- predictor
def _save_video_checkpoint(path, class_names=None, num_classes=3):
    model = Video3DCNN(num_classes=num_classes, model_name="r3d_18", pretrained=False)
    ckpt = {"model_state_dict": model.state_dict(), "config": model.get_config()}
    if class_names is not None:
        ckpt["class_names"] = class_names
    torch.save(ckpt, path)


def test_video_predictor_uses_checkpoint_class_names(tmp_path, video_file):
    from hac.video.inference.predictor import VideoPredictor

    ckpt = tmp_path / "v.pth"
    _save_video_checkpoint(ckpt, class_names=VIDEO_CLASSES)
    predictor = VideoPredictor(ckpt, num_frames=4, device="cpu")
    assert predictor.class_names == VIDEO_CLASSES

    result = predictor.predict_video(video_file, top_k=10)  # clamped to 3 classes
    assert len(result["predictions"]) == 3
    assert result["top_class"] in VIDEO_CLASSES
    total = sum(p["confidence"] for p in result["predictions"])
    assert total == pytest.approx(1.0, abs=1e-4)


def test_video_predictor_fallbacks(tmp_path):
    from hac.video.inference.predictor import VideoPredictor

    ckpt = tmp_path / "v.pth"
    _save_video_checkpoint(ckpt, num_classes=51)  # no stored names -> HMDB51 list
    predictor = VideoPredictor(ckpt, device="cpu")
    assert predictor.class_names[0] == "brush_hair" and len(predictor.class_names) == 51

    with pytest.raises(ValueError):
        VideoPredictor(ckpt, device="cpu", class_names=["only", "two"])


# ------------------------------------------------------------------------ trainer
def test_video_trainer_one_epoch_saves_class_names(tmp_path, frames_dataset_dir):
    from hac.video.training.train import VideoTrainer

    train = VideoDataset(
        frames_dataset_dir / "train", num_frames=4, transform=_frame_transform()
    )
    val = VideoDataset(
        frames_dataset_dir / "test",
        num_frames=4,
        transform=_frame_transform(),
        class_names=train.classes,
    )
    model = Video3DCNN(num_classes=3, model_name="r3d_18", pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = VideoTrainer(
        model=model,
        train_loader=DataLoader(train, batch_size=2),
        val_loader=DataLoader(val, batch_size=2),
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        max_epochs=1,
        use_augmentation=True,
        class_names=train.classes,
    )
    trainer.train()

    ckpt = torch.load(tmp_path / "out" / "latest.pth", weights_only=False)
    assert ckpt["class_names"] == VIDEO_CLASSES
    assert (tmp_path / "out" / "history.json").exists()
