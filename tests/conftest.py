"""Shared fixtures: tiny synthetic datasets and clips so tests run on CPU offline."""

import cv2
import numpy as np
import pytest

IMAGE_CLASSES = ["alpha", "beta", "gamma"]
VIDEO_CLASSES = ["Archery", "Biking", "Typing"]


def _random_image(rng, size=(40, 48)):
    return rng.integers(0, 255, (*size, 3), dtype=np.uint8)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def image_dataset_dir(tmp_path, rng):
    """root/{train,test}/<class>/*.jpg with 2 images per class per split."""
    root = tmp_path / "images"
    for split in ("train", "test"):
        for c in IMAGE_CLASSES:
            d = root / split / c
            d.mkdir(parents=True)
            for i in range(2):
                cv2.imwrite(str(d / f"{i}.jpg"), _random_image(rng))
    return root


@pytest.fixture
def frames_dataset_dir(tmp_path, rng):
    """root/{train,test}/<class>/<clip>/frame_XXXXXX.jpg (8 frames per clip)."""
    root = tmp_path / "frames"
    for split in ("train", "test"):
        for c in VIDEO_CLASSES:
            clip = root / split / c / f"v_{c}_g01_c01"
            clip.mkdir(parents=True)
            for t in range(8):
                cv2.imwrite(
                    str(clip / f"frame_{t:06d}.jpg"), _random_image(rng, (36, 48))
                )
    return root


@pytest.fixture
def video_file(tmp_path, rng):
    """A short MJPG .avi clip (12 frames, 64x48)."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened(), "OpenCV cannot write MJPG .avi in this environment"
    for _ in range(12):
        writer.write(_random_image(rng, (48, 64)))
    writer.release()
    return path
