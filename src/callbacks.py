###############################################################################
# MIT License

# Copyright (c) 2021 Saumya Kumaar Saksena

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
###############################################################################

import os

# Suppress tensorflow notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

from pathlib import Path

import keras
import tensorflow as tf


def get_list_of_callbacks(
    *,
    logdir: Path,
    monitor_metric: str = "val_accuracy",
    mode: str = "max",
    lr_reduction_factor: float = 0.9,
    patience: int = 50,
    min_lr: float = 1e-6,
) -> list:
    callbacks_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=lr_reduction_factor,
        patience=patience,
        min_lr=min_lr,
    )
    callbacks_list.append(reduce_lr)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    callbacks_list.append(tensorboard)
    save_every_nth_epoch = keras.callbacks.ModelCheckpoint(
        logdir / "last.weights.h5",
        monitor=monitor_metric,
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode=mode,
    )
    callbacks_list.append(save_every_nth_epoch)
    best_weights = keras.callbacks.ModelCheckpoint(
        logdir / "best.weights.h5",
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=mode,
    )
    callbacks_list.append(best_weights)
    return callbacks_list
