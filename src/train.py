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

from src.dataloader import Dataloader

# Suppress tensorflow notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import sys
from pathlib import Path

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from src.callbacks import get_list_of_callbacks
from src.utils import get_rich_console, set_seeds


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def training_pipeline(config: DictConfig):

    set_seeds(config.general.seed)
    console = get_rich_console()
    console.print(OmegaConf.to_yaml(config), style="warning")

    console.log(
        "Num GPUs Available: ",
        len(tf.config.experimental.list_physical_devices("GPU")),
        style="warning",
    )

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    dl_train = Dataloader(
        dataset_path=Path(config.general.dataset_path),
        image_splits_path=Path(config.general.image_splits_path),
        class_names_dict=config.class_names,
        input_shape=config.dataloader.input_shape,
        normalize=config.dataloader.normalize,
        mode="train",
    )
    train_generator = dl_train.get_batched_dataset(batch_size=config.general.batch_size)

    dl_val = Dataloader(
        dataset_path=Path(config.general.dataset_path),
        image_splits_path=Path(config.general.image_splits_path),
        class_names_dict=config.class_names,
        input_shape=config.dataloader.input_shape,
        normalize=config.dataloader.normalize,
        mode="val",
    )
    val_generator = dl_val.get_batched_dataset(batch_size=config.general.batch_size)

    dl_test = Dataloader(
        dataset_path=Path(config.general.dataset_path),
        image_splits_path=Path(config.general.image_splits_path),
        class_names_dict=config.class_names,
        input_shape=config.dataloader.input_shape,
        normalize=config.dataloader.normalize,
        mode="test",
    )
    test_generator = dl_test.get_batched_dataset(batch_size=config.general.batch_size)

    if config.general.output_dir is not None:
        experiment_path = (
            Path(config.general.output_dir) / config.general.experiment_name
        )
        Path.mkdir(experiment_path, exist_ok=True, parents=True)
    callbacks = get_list_of_callbacks(
        logdir=experiment_path,
        monitor_metric=config.callbacks.monitor_metric,
        mode=config.callbacks.mode,
        lr_reduction_factor=config.callbacks.lr_reduction_factor,
        lr_reduction_patience=config.callbacks.lr_reduction_patience,
        early_stopping_patience=config.callbacks.early_stopping_patience,
        min_delta=config.callbacks.min_delta,
        min_lr=config.callbacks.min_lr,
    )

    def create_model():
        base_model = hydra.utils.instantiate(config.backbones.backbone)
        # Freeze the base model
        base_model.trainable = config.general.freeze_backbone
        # Add custom layers for transfer learning
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        output = tf.keras.layers.Dense(
            config.dataloader.num_classes, activation="softmax"
        )(x)
        # Create the final model
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

        optimizer = hydra.utils.instantiate(config.optimizer.optimizer_adam)
        loss = hydra.utils.instantiate(config.loss.sparse_categorical_crossentropy)
        metrics = [hydra.utils.instantiate(config.metrics.sparse_categorical_accuracy)]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model

    model = create_model()
    model.summary()

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.general.num_epochs,
        steps_per_epoch=dl_train.length // config.general.batch_size,
        validation_steps=dl_val.length // config.general.batch_size,
        callbacks=callbacks,
    )

    metrics = model.evaluate(
        test_generator,
        batch_size=config.general.batch_size,
        steps=len(test_generator),
        return_dict=True,
    )
    console.print(metrics)


if __name__ == "__main__":
    # This hack is required to prevent hydra from creating output directories.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # Disable Pylint warning about missing parameter, config is passed by Hydra.
    # pylint: disable=no-value-for-parameter
    training_pipeline()
