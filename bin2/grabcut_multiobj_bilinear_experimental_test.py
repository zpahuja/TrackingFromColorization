import os
import sys
import time

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

from tracking_via_colorization.utils.image_process import ImageProcess
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.feeder.dataset import Kinetics, Davis
from tracking_via_colorization.config import Config
from tracking_via_colorization.utils.devices import Devices

palette_path = '/home/nfs/zpahuja2/tracking_from_colorization/eval/davisvideochallenge/davis-2017/data/palette.txt'
color_palette = np.loadtxt(palette_path, dtype=np.uint8).reshape(-1, 3)
palette = color_palette.ravel()

num_segments = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow': 1, 'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1,
                'gold-fish': 4, 'horsejump-high': 2, 'india': 3, 'judo': 2, 'kite-surf': 2, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2, 'motocross-jump': 2, 'paragliding-launch': 2, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 2, 'soapbox': 2}


def get_image(image_arr):
    im = Image.fromarray(image_arr)
    im.putpalette(palette)
    return im


def save_image(im, dirname, frame):
    im.save('%s/%05d.png' % (dirname, frame), format='PNG')


def write_image(image, dirname, frame):
    im = get_image(image)
    save_image(im, dirname, frame)


def overlay(image, indexed_prediction, mask=None, alpha=0.4, square=True):
    im = get_image(indexed_prediction)
    predicted_fg = np.array(im.convert())  # indexed png to RGB
    predicted_fg = np.flip(predicted_fg, 2)  # RGB to BGR

    # increase contrast
    normalizer = np.linalg.norm(predicted_fg, axis=2, ord=np.inf)
    predicted_fg = np.uint8(255 * (predicted_fg / np.atleast_3d(normalizer)))

    if mask is None:
        mask = np.zeros_like(indexed_prediction)
        mask[indexed_prediction != 0] = 255

    mask[mask==1] = 255
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(image, image, mask=mask_inv)

    predicted_fg = cv2.bitwise_and(predicted_fg, predicted_fg, mask = mask)
    image_fg = cv2.bitwise_and(image, image, mask = mask)
    fg = cv2.addWeighted(image_fg, alpha, predicted_fg, 1-alpha, 0)

    overlayed_image = cv2.add(fg, bg)

    if square:
        h, w = overlayed_image.shape[:2]
        overlayed_image = cv2.resize(overlayed_image, (h, h))

    return overlayed_image


def label_image(image, mask, name):
    h, w = image.shape[:2]
    segments = np.where(mask != 0)
    masked_pred = image[segments]

    indexed_prediction = np.zeros((h, w), dtype=np.uint8)
    if len(masked_pred) == 0:
        return indexed_prediction

    kmeans = KMeans(
        n_clusters=num_segments[name], random_state=0).fit(masked_pred)

    # sort labels into consistent order for each frame
    centers = kmeans.cluster_centers_

    for i in range(3):
        centers[:, i] **= (2*i+1)

    idx = np.argsort(centers.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(num_segments[name])

    cluster_labels = lut[kmeans.labels_]

    # cluster_labels = kmeans.labels_

    for idx, label in zip(np.stack(segments, axis=1), cluster_labels):
        i, j = idx
        indexed_prediction[i, j] = np.uint8(label + 1)

    return indexed_prediction


def write_labelled_image(image, mask, dirname, name, frame):
    labelled_image = label_image(image, mask, name)
    write_image(labelled_image, dirname, frame)


def create_directories(dirlist, output_dir, name):
    dirs = {}
    for dir in dirlist:
        dirpath = os.path.join(output_dir, dir, name)
        dirs[dir] = dirpath
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    return dirs


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

        predicted = cv2.resize(predicted, (width, height))
        mfi_predicted = cv2.resize(mfi_predicted, (width, height))

        # perform grabcut
        grabcut_mask = grabcut(image, np.copy(mfi_mask), args.k1, args.k2)  # need binary mask
        grabcut_predicted = cv2.bitwise_and(
            predicted, predicted, mask=grabcut_mask)

        # masking
        _, premfi_mask = cv2.threshold(cv2.cvtColor(
            predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

        _, orig_mask = cv2.threshold(cv2.cvtColor(
            curr['color'], cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

        annotation = label_image(curr['color'], orig_mask, name)
        annotation = cv2.resize(annotation, (width, height))
        labelled_predicted = label_image(predicted, premfi_mask, name)
        mfi_predicted = label_image(mfi_predicted, mfi_mask, name)
        grabcut_predicted = label_image(grabcut_predicted, grabcut_mask, name)

        annotated_image = overlay(image, annotation, square=False)
        predicted_image = overlay(image, labelled_predicted, mask=premfi_mask, square=False)
        mfi_predicted_image = overlay(image, mfi_predicted, mask=mfi_mask, square=False)
        grabcut_predicted_image = overlay(image, grabcut_predicted, mask=grabcut_mask, square=False)
        spacing = np.ones((height, 25, 3), dtype=np.uint8) * 255

        stacked = np.concatenate(
            [annotated_image, spacing, predicted_image, spacing, mfi_predicted_image, spacing, grabcut_predicted_image], axis=1)

        # overlayed on gray scale image
        gray3d = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (width, height))
        annotated_image = overlay(gray3d, annotation)
        predicted_image = overlay(gray3d, labelled_predicted, mask=premfi_mask)
        mfi_predicted_image = overlay(gray3d, mfi_predicted, mask=mfi_mask)
        grabcut_predicted_image = overlay(gray3d, grabcut_predicted, mask=grabcut_mask)

        stacked_gray = np.concatenate(
            [annotated_image, spacing, spacing, predicted_image, spacing, mfi_predicted_image, spacing, grabcut_predicted_image], axis=1)

        # (stacked, premfi_mask, predicted, mfi_mask, mfi_predicted, grabcut_mask, grabcut_predicted)

        dirlist = ['stacked', 'stacked_gray', 'premfi_mask', 'premfi_predicted', 'predicted', 'mfi_mask', 'mfi_predicted', 'grabcut_mask', 'grabcut_predicted']
        dirs = create_directories(dirlist, output_dir, name)

        cv2.imwrite('%s/%05d.jpg' % (dirs['stacked'], frame), stacked)
        cv2.imwrite('%s/%05d.jpg' % (dirs['stacked_gray'], frame), stacked_gray)
        cv2.imwrite('%s/%05d.jpg' % (dirs['predicted'], frame), predicted)

        write_image(premfi_mask, dirs['premfi_mask'], frame)
        write_image(mfi_mask, dirs['mfi_mask'], frame)
        write_image(grabcut_mask, dirs['grabcut_mask'], frame)

        write_image(labelled_predicted, dirs['premfi_predicted'], frame)
        write_image(mfi_predicted, dirs['mfi_predicted'], frame)
        write_image(grabcut_predicted, dirs['grabcut_predicted'], frame)


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
    parser.add_argument('-k1', type=int, default=30)
    parser.add_argument('-k2', type=int, default=50)
    parser.add_argument('-d', '--direction', type=str, default='backward',
                        help='[forward|backward] backward is default')

    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='results')

    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
