"""Regression tests for class-name ordering and cross-split label consistency."""

import numpy as np
import pytest

from hac.common.class_names import (
    HMDB51_CLASSES,
    STANFORD40_CLASSES,
    UCF101_CLASSES,
    get_class_names,
    resolve_class_names,
)


@pytest.mark.parametrize(
    "names,count",
    [(STANFORD40_CLASSES, 40), (UCF101_CLASSES, 101), (HMDB51_CLASSES, 51)],
)
def test_lists_match_loader_order(names, count):
    # Dataset loaders index classes by sorted() directory name; the canonical
    # lists must use the same order or predictions get mislabeled.
    assert len(names) == count
    assert len(set(names)) == count
    assert names == sorted(names)


def test_known_previously_misordered_entries():
    assert STANFORD40_CLASSES[18] == "phoning"
    assert UCF101_CLASSES[34:36] == ["HammerThrow", "Hammering"]
    assert UCF101_CLASSES[46:48] == ["JumpRope", "JumpingJack"]


def test_get_class_names_returns_copy():
    names = get_class_names("UCF101")
    names.append("x")
    assert len(UCF101_CLASSES) == 101
    with pytest.raises(ValueError):
        get_class_names("kinetics")


def test_resolve_class_names_priority():
    stored = [f"c{i}" for i in range(3)]
    assert resolve_class_names({"class_names": stored}, 3, ["a", "b", "c"]) == stored
    assert resolve_class_names({}, 3, ["a", "b", "c"]) == ["a", "b", "c"]
    # Mismatched lengths fall through to generic labels
    assert resolve_class_names({"class_names": ["x"]}, 2, None) == [
        "class_0",
        "class_1",
    ]


def _make_image_split(root, split, classes):
    import cv2

    for c in classes:
        d = root / split / c
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "img.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))


def test_image_dataset_shared_class_list(tmp_path):
    from hac.image.data.dataset import ActionDataset

    _make_image_split(tmp_path, "train", ["a", "b", "c"])
    _make_image_split(tmp_path, "test", ["a", "c"])  # "b" missing in test

    train = ActionDataset(tmp_path, split="train")
    test = ActionDataset(tmp_path, split="test", class_names=train.class_names)
    labels = sorted(label for _, label in test.samples)
    assert labels == [0, 2]  # "c" keeps index 2, not shifted to 1


def test_video_dataset_shared_class_list(tmp_path):
    import cv2

    from hac.video.data.dataset import VideoDataset

    for split, classes in [("train", ["a", "b", "c"]), ("test", ["a", "c"])]:
        for c in classes:
            clip = tmp_path / split / c / "clip0"
            clip.mkdir(parents=True)
            for t in range(4):
                cv2.imwrite(
                    str(clip / f"frame_{t:06d}.jpg"), np.zeros((8, 8, 3), np.uint8)
                )

    train = VideoDataset(tmp_path / "train", num_frames=4)
    test = VideoDataset(tmp_path / "test", num_frames=4, class_names=train.classes)
    assert sorted(label for _, label in test.samples) == [0, 2]
