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

# from sklearn.utils import shuffle
from tensorflow.python.keras import backend as K

# import tensorflow as tf
from src.utils import get_rich_console

if K == "tensorflow":
    K.set_image_data_format("channels_last")


class Dataloader:
    def __init__(self, class_names_dict: dict):
        self.dataset_path = Path("/data/saumya/Datasets/Stanford40/JPEGImages/")
        self.train_path = Path(
            "/data/saumya/Datasets/Stanford40/ImageSplits/global_train.txt"
        )
        self.val_path = Path(
            "/data/saumya/Datasets/Stanford40/ImageSplits/global_val.txt"
        )
        self.console = get_rich_console()
        self._locate_dataset_files()
        self.class_names_dict = class_names_dict

    # The `_locate_dataset_files` method in the `Dataloader` class is responsible
    # for locating the dataset files specified in the `train_path` and `val_path`
    # attributes. Here is a breakdown of what the method is doing:
    def _locate_dataset_files(self):
        with open(self.train_path, "r") as f:
            train_list = f.readlines()
        with open(self.val_path, "r") as f:
            val_list = f.readlines()

        train_list = [item.replace("\n", "") for item in train_list]
        val_list = [item.replace("\n", "") for item in val_list]
        # The `existence_check_train` list comprehension is checking for the existence
        # of each file in the `train_list` within the specified dataset path.
        # It iterates over each item in the`train_list`, constructs the
        # full path by appending it to the dataset path, and then checks
        # if that file exists using the `exists()` method from the `Path`
        # class. The result is a list of boolean values indicating
        # whether each file in the `train_list` exists in the dataset path or not.
        existence_check_train = [
            (self.dataset_path / item).exists() for item in train_list
        ]
        existence_check_val = [(self.dataset_path / item).exists() for item in val_list]
        if all(existence_check_train):
            self.console.print(
                f"Found {len(existence_check_train)} files for training!"
            )
        if all(existence_check_val):
            self.console.print(
                f"Found {len(existence_check_val)} files for validation!"
            )


if __name__ == "__main__":
    data = Dataloader()
