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

import random
import sys
from pathlib import Path

import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from src.utils import get_rich_console, set_seeds


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def split_train_val(config: DictConfig):

    set_seeds(config.general.seed)
    console = get_rich_console()
    console.print(OmegaConf.to_yaml(config), style="warning")

    base_path = Path(config.general.image_splits_path)
    global_train_split = []
    global_val_split = []
    for class_name, index in track(
        config.class_names.items(), description="Analyzing..."
    ):
        sample = base_path / f"{class_name}_train.txt"

        with open(sample, "r") as f:
            data = f.readlines()

        data = [item.replace("\n", "") for item in data]
        random.shuffle(data)
        data_train = data[: int(config.general.train_val_split_ratio * len(data))]
        console.print(
            f"Found {len(data)} total for {class_name}. Train samples"
            f" {len(data_train)}!"
        )
        for item in data:
            if item in data_train:
                global_train_split.append(item)
            else:
                global_val_split.append(item)


if __name__ == "__main__":
    # This hack is required to prevent hydra from creating output directories.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # Disable Pylint warning about missing parameter, config is passed by Hydra.
    # pylint: disable=no-value-for-parameter
    split_train_val()
