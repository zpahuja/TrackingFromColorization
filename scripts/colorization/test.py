"""
TODO.

TODO haven't changed types in data structure to correspond to int32, int32
scaled length could make image in floats
"""
import os
import sys
import time
import copy
import argparse
import logging
import cv2
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

sys.path.append('.')
sys.path.append('/home/nfs/zpahuja2/tracking_from_colorization')  # vision clus

from colorizer import model
from colorizer.dataset import Kinetics
from colorizer.dataset import Davis
from colorizer.config import Config
from colorizer.utils import Devices
from colorizer.utils import ImageProcessor


FILE_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
LOGGER = logging.getLogger(__name__)


def dataflow(name='davis', scale=1):
    """Compute graph to retrieve index, grayscale index, annotation."""
    cfg = Config.get_instance()

    # get test index one at a time
    if name == 'davis':
        data_dirpath = cfg['data_dir']['davis']
        data = Davis(data_dirpath, num_frames=1, shuffle=False)
    elif name == 'kinetics':
        data_dirpath = cfg['data_dir']['kinetics']
        data = Kinetics(data_dirpath, num_frames=1, skips=[0], shuffle=False)
    else:
        raise Exception('Dataset [%s] not supported.' % name)

    # repeat Kinetics index since Davis has image and annotated frames
    if name != 'davis':
        data = df.MapData(data, lambda dp: [dp[0], dp[1], dp[1]])

    data = df.MapData(data, lambda dp: [dp[0], dp[1], dp[2]])
    length = 256 * scale
    size = (length, length)

    # resize frames to 256x256
    data = df.MapDataComponent(
        data, ImageProcessor.resize(small_axis=length), index=1)
    data = df.MapDataComponent(
        data, lambda images: cv2.resize(images[0], size), index=2)

    # get index, original index, gray scale index, annotation mask
    data = df.MapData(data, lambda dp: [
        dp[0],
        dp[1][0],
        cv2.cvtColor(cv2.resize(dp[1][0], size),
                     cv2.COLOR_BGR2GRAY).reshape((length, length, 1)),
        dp[2],
    ])
    data = df.MultiProcessPrefetchData(data, nr_prefetch=32, nr_proc=1)
    return data


def main(args):
    cfg = Config(args.config) if args.config else Config()
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s',
                    args, cfg, device_info)

    scale = args.scale
    image_len, label_len = 256 * scale, 32 * scale
    data = dataflow(args.name, scale)
    data.reset_state()

    num_inputs = args.num_ref_frames + 1  # WHY? TODO
    placeholders = {
        'features': tf.placeholder(tf.int32, (None, num_inputs, image_len, image_len, 1), 'features'),
        'labels': tf.placeholder(tf.int32, (None, num_inputs, label_len, label_len, 1), 'labels'),
    }
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer()
    hparams = tf.contrib.training.HParams(**hparams)

    estimator_spec = model.Colorizer.get('resnet', model.ResNetColorizer, num_ref_frames=args.num_ref_frames, predict_direction=args.direction)(
        features=placeholders['features'],
        labels=placeholders['labels'],
        mode=tf.estimator.ModeKeys.PREDICT,
        params=hparams,
    )

    session = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(session, args.checkpoint)

    # TODO change zeros
    dummy_labels = np.zeros(
        (1, num_inputs, label_len, label_len, 1), dtype=np.int32)

    num_images, video_index = 0, -1
    start_time = time.time()  # TODO replace with SplitTimer

    for idx, image, gray, color in data.get_data():
        curr = {'image': image, 'gray': gray, 'color': color}
        num_images += 1

        if idx == 0:
            tf.logging.info('Avg elapsed time per image: %.3f seconds',
                            (time.time() - start_time) / num_images)
            start_time = time.time()
            num_images = 0
            video_index += 1
            dummy_features = [np.zeros(
                (image_len, image_len, 1), dtype=np.int32) for _ in range(num_inputs)]
            dummy_references = [np.zeros(
                (image_len, image_len, 3), dtype=np.int32) for _ in range(args.num_ref_frames)]

            prev = copy.deepcopy(curr)
            dummy_features = dummy_features[1:] + [prev['gray']]
            tf.logging.info('Video index: %04d', video_index)

        # revise grayscale features and references
        if idx <= args.num_ref_frames:
            dummy_features = dummy_features[1:] + [curr['gray']]
            dummy_references = dummy_references[1:] + [curr['color']]

        features = np.expand_dims(
            np.stack(dummy_features[1:] + [curr['gray']], axis=0), axis=0)
        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: features,
            placeholders['labels']: dummy_labels,
        })

        # predict color
        matrix_size = label_len**2
        indices = np.argmax(predictions['similarity'], axis=-1).reshape((-1,))
        mapping = np.zeros((matrix_size, 2))
        for i, index in enumerate(indices):
            f = (index // matrix_size) % args.num_ref_frames
            y = index // label_len
            x = index % label_len
            mapping[i, :] = [x, (args.num_ref_frames - f - 1) * label_len + y]

        mapping = np.array(mapping, dtype=np.float32).reshape(
            (label_len, label_len, 2))

        height, width = mapping.shape[:2]
        reference_colors = np.concatenate(dummy_references, axis=0)

        predicted = cv2.remap(cv2.resize(
            reference_colors, (width, height * args.num_ref_frames)), mapping, None, cv2.INTER_LINEAR)

        predicted = cv2.resize(predicted, (image_len, image_len))
        # curr['color'] = np.copy(predicted)

        height, width = image.shape[:2]
        predicted = cv2.resize(predicted, (width, height))
        prev = copy.deepcopy(curr)

        if args.name == 'davis':
            _, mask = cv2.threshold(cv2.cvtColor(
                predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            predicted = cv2.add(cv2.bitwise_and(
                image, image, mask=mask_inv), predicted)
            predicted = cv2.addWeighted(image, 0.3, predicted, 0.7, 0)

        stacked = np.concatenate([image, predicted], axis=1)
        similarity = (np.copy(predictions['similarity']).reshape(
            (label_len**2 * args.num_ref_frames, -1)) * 255.0).astype(np.int32)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale))
        similarity = cv2.resize(cv2.dilate(
            similarity, kernel), (label_len * 2 * args.num_ref_frames, label_len * 2))
        output_dir = '%s/%04d' % (args.output, video_index)

        for name, result in [('image', stacked), ('similarity', similarity)]:
            folder = os.path.join(output_dir, name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite('%s/%04d.jpg' % (folder, idx), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0, 1, 2, 3])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-s', '--scale', type=int, default=1)
    parser.add_argument('-n', '--num-ref-frames', type=int, default=1)
    parser.add_argument('-d', '--direction', type=str, default='backward',
                        help='[forward|backward] backward is default')
    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='out/')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(args)
