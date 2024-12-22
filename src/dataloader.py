###############################################################################
# MIT License

# Copyright (c) 2025 Saumya Kumaar Saksena

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################

import os

# Suppress tensorflow notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

# import json
from pathlib import Path
from typing import Optional

import tensorflow as tf

# from sklearn.utils import shuffle
from tensorflow.python.keras import backend as K

from src.utils import get_rich_console

if K == "tensorflow":
    K.set_image_data_format("channels_last")


class Dataloader:
    def __init__(
        self,
        dataset_path: Path,
        image_splits_path: Path,
        class_names_dict: dict,
        mode: str,
        input_shape: tuple,
        debug: Optional[bool] = False,
        normalize: Optional[bool] = False,
    ):
        self.class_names_dict = class_names_dict
        self.mode = mode
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.filepath = image_splits_path / f"global_{self.mode}.txt"
        self.console = get_rich_console()
        self.files_path_list, self.gt_labels = self._locate_dataset_files()
        self.debug = debug
        self.normalize = normalize

    def _locate_dataset_files(self) -> tuple[list, list]:
        with open(self.filepath, "r") as f:
            files_list = f.readlines()

        files_list = [item.replace("\n", "") for item in files_list]
        self.length = len(files_list)

        existence_check = [(self.dataset_path / item).exists() for item in files_list]
        if all(existence_check):
            self.console.print(
                f"Found {len(existence_check)} files for mode: {self.mode}!"
            )

        files_path_list = [str(self.dataset_path / item) for item in files_list]
        gt_labels = []
        for item in files_list:
            suffix = item.split("_")[-1]
            item = item.replace("_" + suffix, "")
            gt_labels.append(item)
        gt_labels = [self.class_names_dict[item] for item in gt_labels]
        return files_path_list, gt_labels

    def _random_crop(self, tensor: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        tensor = tf.image.stateless_random_crop(tensor, self.input_shape, seed)
        return tensor

    def _random_flips(self, tensor: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        tensor = tf.image.stateless_random_flip_left_right(tensor, seed)
        tensor = tf.image.stateless_random_flip_up_down(tensor, seed)
        return tensor

    def _random_cast(self, tensor: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        tensor = tf.image.stateless_random_contrast(tensor, 0.4, 1.6, seed)
        return tensor

    def parse_data(self, items, seed):
        # Read images from disk
        rgb_name, gt = items[0], items[1]
        rgb = tf.io.decode_png(tf.io.read_file(rgb_name), channels=3)

        # Set seeds for augmentation
        # seed_cropping = tf.random.stateless_uniform(
        #     [2], minval=0, maxval=1000, dtype=tf.int32, seed=(seed[0], seed[0] + 1)
        # )
        seed_flipping = tf.random.stateless_uniform(
            [2], minval=0, maxval=1000, dtype=tf.int32, seed=(seed[0] + 2, seed[0] + 3)
        )
        seed_cast = tf.random.stateless_uniform(
            [2], minval=0, maxval=1000, dtype=tf.int32, seed=(seed[0] + 4, seed[0] + 5)
        )

        # Crop randomly to appropriate shape
        # rgb = self._random_crop(rgb, seed_cropping)
        rgb = tf.image.resize(rgb, (self.input_shape[0], self.input_shape[1]))

        if self.mode == "train":
            # Flip and cast only during training
            rgb = self._random_flips(rgb, seed_flipping)
            rgb = self._random_cast(rgb, seed_cast)

        # Convert to tf.float32
        rgb = tf.cast(rgb, dtype=tf.float32)
        gt = tf.cast(gt, dtype=tf.int64)

        # Normalize
        if self.normalize:
            rgb = rgb / 255.0

        if self.debug:
            return (rgb_name, rgb), gt
        else:
            return rgb, gt

    def get_batched_dataset(self, batch_size):
        seeds = tf.data.Dataset.random(seed=0).batch(batch_size)
        auto = tf.data.experimental.AUTOTUNE
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.files_path_list, self.gt_labels)
        )
        self.dataset = tf.data.Dataset.zip((self.dataset, seeds))
        self.dataset = self.dataset.shuffle(
            buffer_size=len(self.files_path_list), reshuffle_each_iteration=True
        )
        self.dataset = (
            self.dataset.repeat()
            .map(map_func=self.parse_data, num_parallel_calls=auto)
            .batch(batch_size=batch_size)
        )
        return self.dataset


if __name__ == "__main__":
    import os

    import hydra
    from hydra import compose, initialize
    from rich.progress import track

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../configs")

    console = get_rich_console()
    config = compose(config_name="train")

    dl_example = Dataloader(
        dataset_path=config.general.dataset_path,
        image_splits_path=config.general.image_splits_path,
        class_names_dict=config.class_names,
        mode="train",
    )
    generator = dl_example.get_batched_dataset(1)
    sample = generator.take(10)
    sample = track(
        sample,
        description="Generating samples...",
        transient=False,
        console=get_rich_console(),
    )
    for _, items in enumerate(sample):
        print(tf.shape(items[0]), tf.shape(items[1]))
        print(items[1])
