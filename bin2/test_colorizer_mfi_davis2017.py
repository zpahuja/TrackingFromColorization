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

num_segments = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow': 1, 'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 4, 'horsejump-high': 2, 'india': 3, 'judo': 2, 'kite-surf': 2, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2, 'motocross-jump': 2, 'paragliding-launch': 2, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 2, 'soapbox': 2}


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
        dp[1][0], # original small axis 256
        cv2.cvtColor(cv2.resize(dp[1][0], size), cv2.COLOR_BGR2GRAY).reshape((size[0], size[1], 1)), # gray 256x256
        dp[2], # annotated mask
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
        # tf.logging.info('video name: %s, frame index: %d', name, frame)

        if frame == 0:
            if video_index != -1:
                tf.logging.info('video name: %s, video index: %04d', name, video_index)
                tf.logging.info('avg elapsed time per image: %.3fsec', (time.time() - start_time) / num_images)
            start_time = time.time()
            num_images = 0
            video_index += 1
            dummy_features = [np.zeros((256 * scale, 256 * scale, 1), dtype=np.float32) for _ in range(num_inputs)]
            dummy_references = [np.zeros((256 * scale, 256 * scale, 3), dtype=np.uint8) for _ in range(args.num_reference)]
            dummy_features = dummy_features[1:] + [curr['gray']]
            prev_prediction = curr['color']

        if frame <= args.num_reference:
            dummy_features = dummy_features[1:] + [curr['gray']]
            dummy_references = dummy_references[1:] + [curr['color']]

        features = np.expand_dims(np.stack(dummy_features[1:] + [curr['gray']], axis=0), axis=0)
        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: features,
            placeholders['labels']: dummy_labels,
        })

        # find mapping from similarity_matrix to frame pixel
        matrix_size = cmap_size * cmap_size * scale * scale
        indices = np.argmax(predictions['similarity'], axis=-1).reshape((-1,))
        mapping = np.zeros((matrix_size, 2))
        for i, index in enumerate(indices):
            f = (index // (matrix_size)) % args.num_reference
            y = index // (cmap_size * scale)
            x = index % (cmap_size * scale)
            mapping[i, :] = [x, (args.num_reference - f - 1) * (cmap_size * scale) + y]
        mapping = np.array(mapping, dtype=np.float32).reshape((cmap_size * scale, cmap_size * scale, 2))

        height, width = mapping.shape[:2]
        reference_colors = np.concatenate(dummy_references, axis=0)
        # reference_labels = np.zeros(reference_colors.shape[:2], dtype=np.uint8)
        #
        # h, w = reference_labels.shape
        # colors = color_palette[:6]
        # for i in range(h):
        #     for j in range(w):
        #         reference_labels[i,j] = np.argmin(np.linalg.norm(reference_colors[i,j] - colors, axis=1))

        predicted = cv2.remap(cv2.resize(reference_colors, (width, height * args.num_reference)), mapping, None, cv2.INTER_LINEAR)
        # predicted_labels = cv2.remap(reference_labels, mapping, None, cv2.INTER_LINEAR)
        # tf.logging.info("predicted shape after remapping: %s", str(predicted.shape))
        # YOU ARE NOT DOING CMAP = 64

        #### MFI
        _, mask = cv2.threshold(cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
        _h, _w = mask.shape
        A = np.asarray(mask).astype(int)
        A[A<128] = -1
        A[A>=128] = 1

        def get_neighbors(i, j, h, w):
            candidates = [(i-1,j), (i+1,j), (i, j-1), (i, j+1)]
            return [(a,b) for a,b in candidates if a>=0 and b>=0 and a<h and b<w]

        threshold = 0.001
        theta = args.theta
        pi = np.zeros((_h, _w))

        while True:
            old_pi = np.copy(pi)
            for i in range(_h):
                for j in range(_w):
                    a = A[i,j]
                    for x,y in get_neighbors(i, j, _h, _w):
                        a += theta * (2 * pi[x,y] - 1)
                    b = -a
                    a, b = np.exp(a), np.exp(b)
                    pi[i,j] = np.exp(a)/(np.exp(a) + np.exp(b))

            delta = np.linalg.norm(pi - old_pi)
            if delta < threshold:
                mask[pi>=.5] = 1
                mask[pi<.5] = 0
                break

        # write mask and A
        """
        output_dir = '%s' % (args.output)
        cv2.imwrite('%s1_%s_%05d.jpg' % (output_dir, name, frame), mask)
        cv2.imwrite('%s2_%s_%05d.jpg' % (output_dir, name, frame), A)
        mask = cv2.resize(mask, (_w, _h))
        """
        ### MFI ends here

        predicted_eval = cv2.bitwise_and(predicted, predicted, mask=mask)
        dim = predicted_eval.shape[0]

        # index predicted eval
        # cv2.imwrite('%s/%s_%05d.png' % (args.output, name, frame), predicted_eval)
        _eval = predicted_eval.reshape(-1,3)
        _eval_vec = np.int32(_eval > 0)
        # _colors = np.int32(color_palette[:num_segments[name]+1] > 0)
        _colors = np.int32(color_palette[:8,::-1] > 0)
        m = np.array([[4,2,1]]).T
        binarized = _eval_vec @ m
        for id, items in enumerate(zip(binarized, _eval_vec)):
            bin, e = items
            # print(id//32, id%32, predicted[id//32, id%32], list(e), bin)
        bin_index = _colors @ m
        Y = binarized.reshape(-1)
        X = bin_index.reshape(-1)
        # print("color indices: ", X)

        # print("unique binarized: ", np.unique(binarized))
        # print("binarized NA: ", len([binarized[np.where(np.in1d(Y,X) == False)]]))

        """
        for a in bin_index:
            print("VAL:", a)
            ii = np.where(binarized==a)[0]
            print(len(list(zip(ii//32,ii%32, A[ii//32,ii%32]))))
        """

        binarized[np.where(np.in1d(Y,X) == False)] = 0
        # print("unique binarized after removing NA: ", np.unique(binarized))
        indexed_eval = np.where(Y.reshape(Y.size, 1) == X)[1]
        indexed_eval = indexed_eval.reshape(dim, dim)

        height, width = image.shape[:2]
        # indexed_eval = indexed_eval.reshape(width, height)

        # indexed_eval = cv2.resize(indexed_eval, (256 * scale, 256 * scale), interpolation=cv2.INTER_AREA)
        indexed_eval = cv2.resize(indexed_eval.astype('uint8'), (width, height), interpolation=cv2.INTER_AREA)
        """
        if len(np.unique(indexed_eval)) != num_segments[name] + 1:
            print("ATTENTION unique in indexed_eval: ", np.unique(indexed_eval), "should be size:", num_segments[name] + 1)
        """

        im = Image.fromarray(indexed_eval.astype('uint8'))
        im.putpalette(color_palette.ravel())

        output_dir = '%s' % (args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predicted_eval_dir = os.path.join(args.output, 'mask_2017', name)
        if not os.path.exists(predicted_eval_dir):
            os.makedirs(predicted_eval_dir)

        im.save('%s/%05d.png' % (predicted_eval_dir, frame), format='PNG')

        mask[mask==1] = 255
        # dummy_references = dummy_references[1:] + [predicted] ## TRY CHANGING THIS

        # curr['color'] = np.copy(predicted)
        # I GUESS YOU SHOULD DO MFI, KMEANS HERE
        # predicted shape 256 * 256

        predicted = cv2.resize(predicted, (256 * scale, 256 * scale))
        predicted = cv2.resize(predicted, (width, height)) # small axis 256
        # predicted_labels = cv2.resize(predicted_labels, (width, height))
        # prev = copy.deepcopy(curr)

        if args.name == 'davis':
            mask = cv2.resize(mask, (256 * scale, 256 * scale))
            mask = cv2.resize(mask, (width, height))
            _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
            _, mask_inv_prev = cv2.threshold(cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            predicted_col = predicted
            # m,n,o = predicted_col.shape
            # for i in range(m):
            #     for j in range(n):
            #         if not np.all(predicted_col[i,j,:] == np.zeros(3)):
            #             print(predicted_col[i,j,:])

            # masked_prediction = cv2.bitwise_and(predicted, predicted, mask=mask)
            predicted_prev = cv2.add(cv2.bitwise_and(image, image, mask=mask_inv_prev), predicted)
            predicted_prev = cv2.addWeighted(image, 0.3, predicted_prev, 0.7, 0)

            predicted_eval = cv2.bitwise_and(predicted, predicted, mask=mask)
            predicted = cv2.add(cv2.bitwise_and(image, image, mask=mask_inv), predicted_eval)
            predicted = cv2.addWeighted(image, 0.3, predicted, 0.7, 0)

        stacked = np.concatenate([predicted_prev, predicted], axis=1)
        # stacked = np.concatenate([image, predicted], axis=1)

        # similarity = (np.copy(predictions['similarity']).reshape((cmap_size * cmap_size * scale * scale * args.num_reference, -1)) * 255.0).astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale))
        # similarity = cv2.resize(cv2.dilate(similarity, kernel), (cmap_size * scale * 2 * args.num_reference, cmap_size * scale * 2))

        # for compare, mask, _predicted_eval, _predicted in [(stacked, mask, predicted_eval, predicted)]:
        for compare, _mask in [(stacked, mask)]:
            image_dir = os.path.join(output_dir, 'image', name)
            eval_dir = os.path.join(output_dir, 'mask_2016', name)

            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
                os.makedirs(eval_dir)

            cv2.imwrite('%s/%05d.jpg' % (image_dir, frame), compare)

            im = Image.fromarray(_mask)
            im.putpalette(color_palette.ravel())
            im.save('%s/%05d.png' % (eval_dir, frame), format='PNG')

            # _predicted = cv2.resize(_predicted, (854, 480))
            # tf.logging.info("Indexing PNG...")

            # h, w = _predicted.shape[:2]
            # # print("shape of _predicted", _predicted.shape)
            # # print("shape of mask", mask.shape)
            # segments = np.where(mask != 0)
            #
            # masked_pred = _predicted[segments]
            # # print("shape of masked_pred", masked_pred.shape)
            #
            # kmeans = KMeans(n_clusters=num_segments[name], random_state=0).fit(masked_pred)
            # indexed_prediction = np.zeros((h,w), dtype=np.uint8)
            # # print("index pred shape: ", indexed_prediction.shape)
            #
            # # colors = np.flip(color_palette[1:num_segments[name]+1], axis=1)
            # cluster_labels = kmeans.labels_
            # # print("shape of cluster labels", cluster_labels.shape)
            #
            # for idx, label in zip(np.stack(segments, axis=1), cluster_labels):
            #     i, j = idx
            #     # print("coords:", i, j, "label:", label)
            #     indexed_prediction[i,j] = np.uint8(label + 1)
            #
            # """
            # print("segments len: ", len(segments))
            # print("segments[0] shape:", segments[0].shape)
            # """
            #
            # # ctr = 0
            # # for i, j in segments:
            # #     # print("coords:", idx, "label:", label)
            # #     print(i, j)
            # #     indexed_prediction[i, j] = np.uint8(cluster_labels[ctr])
            # #     ctr += 1
            #
            # im = Image.fromarray(indexed_prediction)
            # im.putpalette(color_palette.ravel())
            # im.save('%s/%05d.png' % (kmean_dir, frame), format='PNG')
            # # tf.logging.info("Finished writing predicted frame %05d", frame)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0,1,2,3])
    parser.add_argument('--checkpoint', type=str, default="models/colorizer2/model.ckpt-64000")
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--cmap', type=int, default=32)
    parser.add_argument('-s', '--scale', type=int, default=1)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('-n', '--num-reference', type=int, default=3)
    parser.add_argument('-t', '--theta', type=float, default=0.4)
    parser.add_argument('-d', '--direction', type=str, default='backward', help='[forward|backward] backward is default')

    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='results')

    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
