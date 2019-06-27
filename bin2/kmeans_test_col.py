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

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tracking_via_colorization.utils.devices import Devices
from tracking_via_colorization.config import Config
from tracking_via_colorization.feeder.dataset import Kinetics, Davis
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer
from tracking_via_colorization.utils.image_process import ImageProcess

palette_path = '/home/nfs/zpahuja2/tracking_from_colorization/eval/davisvideochallenge/davis-2017/data/palette.txt'
color_palette = np.loadtxt(palette_path, dtype=np.uint8).reshape(-1,3)

cnummap = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow': 1, 'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 4, 'horsejump-high': 2, 'india': 3, 'judo': 2, 'kite-surf': 2, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2, 'motocross-jump': 2, 'paragliding-launch': 2, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 2, 'soapbox': 2}


def dataflow(name='davis', scale=1, split='val'):
    if name == 'davis':
        ds = Davis('/data/zubin/videos/davis', name=split, num_frames=1, shuffle=False)
    elif name == 'kinetics':
        ds = Kinetics('/data/public/rw/datasets/videos/kinetics', num_frames=1, skips=[0], shuffle=False)
    else:
        raise Exception('not support dataset %s' % name)

    if name != 'davis':
        ds = df.MapData(ds, lambda dp: [dp[0], dp[1], dp[1]])

    ds = df.MapData(ds, lambda dp: [
        dp[0], # index
        dp[1], # original
        dp[2], # mask
        dp[3], # name
    ])
    size = (256 * scale, 256 * scale)

    ds = df.MapDataComponent(ds, ImageProcess.resize(small_axis=256 * scale), index=1)
    ds = df.MapDataComponent(ds, lambda images: cv2.resize(images[0], size), index=2)

    ds = df.MapData(ds, lambda dp: [
        dp[0], # index
        dp[1][0], # original
        cv2.cvtColor(cv2.resize(dp[1][0], size), cv2.COLOR_BGR2GRAY).reshape((size[0], size[1], 1)), # gray
        dp[2], # mask
        dp[3], # name
    ])
    ds = df.MultiProcessPrefetchData(ds, nr_prefetch=32, nr_proc=1)
    return ds

def main(args):
    Config(args.config)
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s', args, Config.get_instance(), device_info)

    scale = args.scale
    ds = dataflow(args.name, scale, args.split)
    ds.reset_state()

    cmap_size = args.cmap

    num_inputs = args.num_reference + 1
    placeholders = {
        'features': tf.placeholder(tf.float32, (None, num_inputs, 256 * scale, 256 * scale, 1), 'features'),
        'labels': tf.placeholder(tf.int64, (None, num_inputs, cmap_size * scale, cmap_size * scale, 1), 'labels'),
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

    dummy_labels = np.zeros((1, num_inputs, cmap_size * scale, cmap_size * scale, 1), dtype=np.int64)

    video_index = -1
    start_time = time.time()
    num_images = 0
    for frame, image, gray, color, name in ds.get_data():
        curr = {'image': image, 'gray': gray, 'color': color}
        num_images += 1

        if frame == 0:
            tf.logging.info('avg elapsed time per image: %.3fsec', (time.time() - start_time) / num_images)
            start_time = time.time()
            num_images = 0
            video_index += 1
            dummy_features = [np.zeros((256 * scale, 256 * scale, 1), dtype=np.float32) for _ in range(num_inputs)]
            dummy_references = [np.zeros((256 * scale, 256 * scale, 3), dtype=np.uint8) for _ in range(args.num_reference)]
            prev = copy.deepcopy(curr)
            dummy_features = dummy_features[1:] + [prev['gray']]
            tf.logging.info('video name: %s, video index: %04d', name, video_index)

        if frame <= args.num_reference:
            dummy_features = dummy_features[1:] + [curr['gray']]
            dummy_references = dummy_references[1:] + [curr['color']]

        features = np.expand_dims(np.stack(dummy_features[1:] + [curr['gray']], axis=0), axis=0)
        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: features,
            placeholders['labels']: dummy_labels,
        })

        matrix_size = cmap_size * cmap_size * scale * scale
        indicies = np.argmax(predictions['similarity'], axis=-1).reshape((-1,))
        mapping = np.zeros((matrix_size, 2))
        for i, index in enumerate(indicies):
            f = (index // (matrix_size)) % args.num_reference
            y = index // (cmap_size * scale)
            x = index % (cmap_size * scale)
            mapping[i, :] = [x, (args.num_reference - f - 1) * (cmap_size * scale) + y]
        mapping = np.array(mapping, dtype=np.float32).reshape((cmap_size * scale, cmap_size * scale, 2))

        height, width = mapping.shape[:2]
        reference_colors = np.concatenate(dummy_references, axis=0)
        reference_colors = cv2.resize(reference_colors, (width, height * args.num_reference))
        # reference_labels = np.zeros(reference_colors.shape[:2], dtype=np.uint8)
        #
        # h, w = reference_labels.shape
        # colors = color_palette[:6]
        # for i in range(h):
        #     for j in range(w):
        #         reference_labels[i,j] = np.argmin(np.linalg.norm(reference_colors[i,j] - colors, axis=1))

        predicted = cv2.remap(reference_colors, mapping, None, cv2.INTER_LINEAR)
        # predicted_labels = cv2.remap(reference_labels, mapping, None, cv2.INTER_LINEAR)
        # tf.logging.info("predicted shape after remapping: %s", str(predicted.shape))

        predicted = cv2.resize(predicted, (256 * scale, 256 * scale))
        curr['color'] = np.copy(predicted)

        height, width = image.shape[:2]
        predicted = cv2.resize(predicted, (width, height))
        # predicted_labels = cv2.resize(predicted_labels, (width, height))
        prev = copy.deepcopy(curr)

        if args.name == 'davis':
            _, mask = cv2.threshold(cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            predicted_col = predicted
            # m,n,o = predicted_col.shape
            # for i in range(m):
            #     for j in range(n):
            #         if not np.all(predicted_col[i,j,:] == np.zeros(3)):
            #             print(predicted_col[i,j,:])
            predicted = cv2.add(cv2.bitwise_and(image, image, mask=mask_inv), predicted)
            predicted = cv2.addWeighted(image, 0.3, predicted, 0.7, 0)

        stacked = np.concatenate([image, predicted], axis=1)
        # similarity = (np.copy(predictions['similarity']).reshape((cmap_size * cmap_size * scale * scale * args.num_reference, -1)) * 255.0).astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale))
        # similarity = cv2.resize(cv2.dilate(similarity, kernel), (cmap_size * scale * 2 * args.num_reference, cmap_size * scale * 2))

        # output_dir = '%s/%s' % (args.output, name)
        output_dir = '%s' % (args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for _, viz, mask, _predicted in [('image', stacked, mask, predicted_col, )]:
            # folder = os.path.join(output_dir, name)
            image_dir = os.path.join(output_dir, 'image', name)
            mask_dir = os.path.join(output_dir, 'mask', name)
            # predicted_dir = os.path.join(output_dir, 'imwrite_predicted', name)
            kmean_dir = os.path.join(output_dir, 'kmeans_predictions', name)

            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
                os.makedirs(mask_dir)
                # os.makedirs(predicted_dir)
                # os.makedirs(predicted_dir2)
                os.makedirs(kmean_dir)

            cv2.imwrite('%s/%05d.jpg' % (image_dir, frame), viz)
            # cv2.imwrite('%s/%05d.png' % (predicted_dir, frame), _predicted)

            im = Image.fromarray(mask)
            im.putpalette(color_palette.ravel())
            im.save('%s/%05d.png' % (mask_dir, frame), format='PNG')

            # _predicted = cv2.resize(_predicted, (854, 480))
            # tf.logging.info("Indexing PNG...")

            h, w = _predicted.shape[:2]
            # print("shape of _predicted", _predicted.shape)
            # print("shape of mask", mask.shape)
            segments = np.where(mask != 0)

            masked_pred = _predicted[segments]
            # print("shape of masked_pred", masked_pred.shape)

            kmeans = KMeans(n_clusters=cnummap[name], random_state=0).fit(masked_pred)
            indexed_prediction = np.zeros((h,w), dtype=np.uint8)
            # print("index pred shape: ", indexed_prediction.shape)

            # colors = np.flip(color_palette[1:cnummap[name]+1], axis=1)
            cluster_labels = kmeans.labels_
            # print("shape of cluster labels", cluster_labels.shape)

            for idx, label in zip(np.stack(segments, axis=1), cluster_labels):
                i, j = idx
                # print("coords:", i, j, "label:", label)
                indexed_prediction[i,j] = np.uint8(label + 1)

            """
            print("segments len: ", len(segments))
            print("segments[0] shape:", segments[0].shape)
            """

            # ctr = 0
            # for i, j in segments:
            #     # print("coords:", idx, "label:", label)
            #     print(i, j)
            #     indexed_prediction[i, j] = np.uint8(cluster_labels[ctr])
            #     ctr += 1

            im = Image.fromarray(indexed_prediction)
            im.putpalette(color_palette.ravel())
            im.save('%s/%05d.png' % (kmean_dir, frame), format='PNG')
            # tf.logging.info("Finished writing predicted frame %05d", frame)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0,1,2,3])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--cmap', type=int, default=32)
    parser.add_argument('-s', '--scale', type=int, default=1)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('-n', '--num-reference', type=int, default=3)
    parser.add_argument('-d', '--direction', type=str, default='backward', help='[forward|backward] backward is default')

    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='results')

    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
