import logging
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


def build_learning_rate(global_step, batch_size, device_count, steps_per_epoch, initial_learning_rate, warmup=True):
    multiplier = (batch_size * device_count) / 8.0
    initial_learning_rate = initial_learning_rate * multiplier
    boundaries = [int(steps_per_epoch * epoch) for epoch in [20, 30, 40]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3]]

    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    if warmup:
        warmup_iter = float(steps_per_epoch * 5)
        ratio = 1.0 / multiplier
        warmup_ratio = tf.minimum(1.0, (1.0 - ratio) * (tf.cast(global_step, tf.float32) / warmup_iter) ** 2 + ratio)
        learning_rate *= warmup_ratio
    return learning_rate
