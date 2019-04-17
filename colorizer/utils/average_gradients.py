import logging
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


def average_gradients(tower_grads):
    average_grads = []
    for grad in tower_grads:
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g in grad:
            if g is None:
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        average_grads.append(grad)

    return average_grads
