# pylint: disable=I1101
import os
import sys
import time
import copy

import cv2
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

from sklearn.cluster import KMeans

from PIL import Image

import mean_field_inference as mfi
from grabcut import grabcut

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

palette_path = '/home/nfs/zpahuja2/tracking_from_colorization/eval/davisvideochallenge/davis-2017/data/palette.txt'
from tracking_via_colorization.utils.image_process import ImageProcess
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.feeder.dataset import Kinetics, Davis
from tracking_via_colorization.config import Config
from tracking_via_colorization.utils.devices import Devices

color_palette = np.loadtxt(palette_path, dtype=np.uint8).reshape(-1, 3)
palette = color_palette.ravel()

num_segments = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow': 1, 'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1,
                'gold-fish': 4, 'horsejump-high': 2, 'india': 3, 'judo': 2, 'kite-surf': 2, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2, 'motocross-jump': 2, 'paragliding-launch': 2, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 2, 'soapbox': 2}


def write_image(image, dirname, frame):
    im = Image.fromarray(image)
    im.putpalette(palette)
    im.save('%s/%05d.png' % (dirname, frame), format='PNG')


def label_image(image, mask, name):
    h, w = image.shape[:2]
    segments = np.where(mask != 0)
    masked_pred = image[segments]
    kmeans = KMeans(
        n_clusters=num_segments[name], random_state=0).fit(masked_pred)
    indexed_prediction = np.zeros((h, w), dtype=np.uint8)
    centers = kmeans.cluster_centers_

    for i in range(3):
        centers[:, i] **= (i+1)

    idx = np.argsort(centers.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(num_segments[name])

    cluster_labels = lut[kmeans.labels_]

    for idx, label in zip(np.stack(segments, axis=1), cluster_labels):
        i, j = idx
        indexed_prediction[i, j] = np.uint8(label + 1)

    return indexed_prediction


def write_labelled_image(image, mask, dirname, name, frame):
    labelled_image = label_image(image, mask, name)
    write_image(labelled_image, dirname, frame)


def dataflow(name='davis', scale=1, split='val'):
    if name == 'davis':
        ds = Davis('/data/zubin/videos/davis', name=split,
                   num_frames=1, shuffle=False)
    elif name == 'kinetics':
        ds = Kinetics('/data/public/rw/datasets/videos/kinetics',
                      num_frames=1, skips=[0], shuffle=False)
    else:
        raise Exception('not support dataset %s' % name)

    if name != 'davis':
        ds = df.MapData(ds, lambda dp: [dp[0], dp[1], dp[1]])

    ds = df.MapData(ds, lambda dp: [
        dp[0],  # index
        dp[1],  # original
        dp[2],  # mask
        dp[3],  # name
    ])
    feature_size = int(256 * scale)
    size = (feature_size, feature_size)

    ds = df.MapDataComponent(ds, ImageProcess.resize(
        small_axis=feature_size), index=1)
    ds = df.MapDataComponent(
        ds, lambda images: cv2.resize(images[0], size), index=2)

    ds = df.MapData(ds, lambda dp: [
        dp[0],  # index
        dp[1][0],  # original small axis 256 x scale
        cv2.cvtColor(cv2.resize(dp[1][0], size), cv2.COLOR_BGR2GRAY).reshape(
            (size[0], size[1], 1)),  # gray (256xscale)x(256xscale)x1
        dp[2],  # annotated mask 256xscale x 256xscale
        dp[3],  # name
    ])
    ds = df.MultiProcessPrefetchData(ds, nr_prefetch=32, nr_proc=1)
    return ds


def main(args):
    Config(args.config)
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s',
                    args, Config.get_instance(), device_info)

    scale = args.scale
    ds = dataflow(args.name, scale, args.split)
    ds.reset_state()

    cmap_size = int(args.cmap * scale)
    feature_size = int(256 * scale)

    num_inputs = args.num_reference + 1
    placeholders = {
        'features': tf.placeholder(tf.float32, (None, num_inputs, feature_size, feature_size, 1), 'features'),
        'labels': tf.placeholder(tf.int64, (None, num_inputs, cmap_size, cmap_size, 1), 'labels'),
    }
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer()
    hparams = tf.contrib.training.HParams(**hparams)

    estimator_spec = Colorizer.get('resnet', ResNetColorizer, num_reference=args.num_reference, predict_direction=args.direction)(
        features=placeholders['features'],
        labels=placeholders['labels'],
        mode=tf.estimator.ModeKeys.PREDICT,
        params=hparams
    )

    session = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(session, args.checkpoint)

    dummy_labels = np.zeros(
        (1, num_inputs, cmap_size, cmap_size, 1), dtype=np.int64)

    video_index = -1
    start_time = time.time()
    num_images = 0
    output_dir = '%s' % (args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame, image, gray, color, name in ds.get_data():
        curr = {'image': image, 'gray': gray, 'color': color}
        num_images += 1

        if frame == 0:
            if video_index != -1:
                tf.logging.info('avg elapsed time per image: %.3fsec',
                                (time.time() - start_time) / num_images)
            start_time = time.time()
            num_images = 0
            video_index += 1
            dummy_features = [np.zeros(
                (feature_size, feature_size, 1), dtype=np.float32) for _ in range(num_inputs)]
            dummy_references = [np.zeros(
                (feature_size, feature_size, 3), dtype=np.uint8) for _ in range(args.num_reference)]
            dummy_features = dummy_features[1:] + [curr['gray']]
            tf.logging.info('video name: %s, video index: %04d',
                            name, video_index)

        if frame <= args.num_reference:
            dummy_features = dummy_features[1:] + [curr['gray']]
            dummy_references = dummy_references[1:] + [curr['color']]

        features = np.expand_dims(
            np.stack(dummy_features[1:] + [curr['gray']], axis=0), axis=0)
        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: features,
            placeholders['labels']: dummy_labels,
        })

        # find mapping from similarity_matrix to frame pixel
        matrix_size = cmap_size * cmap_size
        indices = np.argmax(predictions['similarity'], axis=-1).reshape((-1,))
        mapping = np.zeros((matrix_size, 2))
        for i, index in enumerate(indices):
            f = (index // (matrix_size)) % args.num_reference
            y = index // cmap_size
            x = index % cmap_size
            mapping[i, :] = [x, (args.num_reference - f - 1) * cmap_size + y]
        mapping = np.array(mapping, dtype=np.float32).reshape(
            (cmap_size, cmap_size, 2))

        height, width = mapping.shape[:2]

        reference_colors = np.concatenate(dummy_references, axis=0)
        reference_colors = cv2.resize(
            reference_colors, (width, height * args.num_reference))
        predicted = cv2.remap(reference_colors, mapping,
                              None, cv2.INTER_LINEAR)

        # get mask
        grayscale_predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grayscale_predicted, 10,
                                255, cv2.THRESH_BINARY)
        mask_shape = mask.shape
        mask_arr = np.asarray(mask).astype(int)
        mask_arr = np.where(mask_arr < 128, 0, 1)

        # label segments
        labelled_image = label_image(predicted, mask_arr, name)

        # multi object mfi
        mfi_predicted = mfi.denoise_image(labelled_image, theta=args.theta)
        mask[mfi_predicted == 0] = 0
        mask[mfi_predicted != 0] = 1

        mfi_predicted = cv2.bitwise_and(predicted, predicted, mask=mask)

        height, width = image.shape[:2]
        mfi_mask = cv2.resize(mask * 255, (width, height))

        # binarized
        _, mfi_mask = cv2.threshold(mfi_mask, 60, 255, cv2.THRESH_BINARY)
        mfi_mask[mfi_mask == 255] = 1

        # perform grabcut
        grabcut_mask = grabcut(image, np.copy(mfi_mask)) # need binary mask

        predicted = cv2.resize(predicted, (width, height))
        mfi_predicted = cv2.resize(mfi_predicted, (width, height))

        if args.name == 'davis':
            _, premfi_mask = cv2.threshold(cv2.cvtColor(
                predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(premfi_mask)
            predicted_image = cv2.add(cv2.bitwise_and(
                image, image, mask=mask_inv), predicted)
            predicted_image = cv2.addWeighted(image, 0.4, predicted_image, 0.6, 0)

            mfi_mask_inv = cv2.bitwise_not(mfi_mask * 255)
            mfi_predicted_image = cv2.add(cv2.bitwise_and(
                image, image, mask=mfi_mask_inv), mfi_predicted)
            mfi_predicted_image = cv2.addWeighted(
                image, 0.3, mfi_predicted_image, 0.7, 0)

            # grabcut
            grabcut_mask_inv = cv2.bitwise_not(grabcut_mask * 255)
            grabcut_predicted = cv2.bitwise_and(
                predicted, predicted, mask=grabcut_mask)

            grabcut_predicted_image = cv2.add(cv2.bitwise_and(
                image, image, mask=grabcut_mask_inv), grabcut_predicted)
            grabcut_predicted_image = cv2.addWeighted(
                image, 0.3, grabcut_predicted_image, 0.7, 0)

        stacked = np.concatenate(
            [image, predicted_image, mfi_predicted_image, grabcut_predicted_image], axis=1)

        # (stacked, premfi_mask, predicted, mfi_mask, mfi_predicted, grabcut_mask, grabcut_predicted)

        stacked_dir = os.path.join(output_dir, 'stacked', name)
        premfi_mask_dir = os.path.join(output_dir, 'premfi_mask', name)
        premfi_predicted_dir = os.path.join(output_dir, 'premfi_predicted', name)
        mfi_mask_dir = os.path.join(output_dir, 'mfi_mask', name)
        mfi_predicted_dir = os.path.join(output_dir, 'mfi_predicted', name)
        grabcut_mask_dir = os.path.join(output_dir, 'grabcut_mask', name)
        grabcut_predicted_dir = os.path.join(output_dir, 'grabcut_predicted', name)

        if not os.path.exists(stacked_dir):
            os.makedirs(stacked_dir)
            os.makedirs(premfi_mask_dir)
            os.makedirs(premfi_predicted_dir)
            os.makedirs(mfi_mask_dir)
            os.makedirs(mfi_predicted_dir)
            os.makedirs(grabcut_mask_dir)
            os.makedirs(grabcut_predicted_dir)

        cv2.imwrite('%s/%05d.jpg' % (stacked_dir, frame), stacked)

        write_image(premfi_mask, premfi_mask_dir, frame)
        write_image(mfi_mask, mfi_mask_dir, frame)
        write_image(grabcut_mask, grabcut_mask_dir, frame)

        write_labelled_image(predicted, premfi_mask, premfi_predicted_dir, name, frame)
        write_labelled_image(mfi_predicted, mfi_mask, mfi_predicted_dir, name, frame)
        write_labelled_image(grabcut_predicted, grabcut_mask, grabcut_predicted_dir, name, frame)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0, 1, 2, 3])
    parser.add_argument('--checkpoint', type=str,
                        default='models/colorizer2/model.ckpt-64000')
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--cmap', type=int, default=32)
    parser.add_argument('-s', '--scale', type=float, default=1)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('-n', '--num-reference', type=int, default=3)
    parser.add_argument('-t', '--theta', type=float, default=0.5)
    parser.add_argument('-d', '--direction', type=str, default='backward',
                        help='[forward|backward] backward is default')

    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='results')

    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
