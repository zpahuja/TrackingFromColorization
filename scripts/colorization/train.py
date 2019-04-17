import os
import sys
import copy
import cv2
import argparse
import logging
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

sys.path.append('.')
sys.path.append('/home/nfs/zpahuja2/tracking_from_colorization')  # vision clus

from colorizer import model
from colorizer.dataset import Kinetics
from colorizer.config import Config
from colorizer.utils import Devices
from colorizer.utils import ImageProcessor


FILE_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
DEFAULT_CENTROIDS_PATH = os.path.join(ROOT_PATH, 'out', 'centroids', 'centroids_16k_kinetics_10000samples.npy')
LOGGER = logging.getLogger(__name__)


def dataflow(centroids, num_ref_frames=3, num_process=16, shuffle=True):
    """
    Compute graph to retrieve 3 reference and 1 target frame from Kinetics dataset,
    Downsample grayscale frames to 256x256 and colorized frames to 32x32 feature maps in Lab space,
    Cluster colors in colorized frames, prepare data in the format (input, output)
    """
    cfg = Config.get_instance()
    kinetics_dirpath = cfg['data_dir']['kinetics']

    # get frame and 3 prior reference frames with certain number of skips
    data = Kinetics(kinetics_dirpath, num_frames=num_ref_frames + 1, skips=[0, 4, 4, 8][:num_ref_frames + 1], shuffle=shuffle)

    # downsample frames to 256x256
    data = df.MapDataComponent(data, ImageProcessor.resize(small_axis=256), index=1)
    data = df.MapDataComponent(data, ImageProcessor.crop(shape=(256, 256)), index=1)
    # data = df.MapDataComponent(data, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=1)

    # split frames into 3 references and 1 target frame, with deep copies of each at odd indices
    data = df.MapData(data, lambda dp: [dp[1][:num_ref_frames], copy.deepcopy(dp[1][:num_ref_frames]), dp[1][num_ref_frames:], copy.deepcopy(dp[1][num_ref_frames:])])

    # decolorize first set of reference and target frame(s)
    for idx in [0, 2]:
        data = df.MapDataComponent(data, lambda images: [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(256, 256, 1) for image in images], index=idx)

    # downsample to 32x32 feature map, keep only 'ab' from Lab color space & find nearest cluster color for every pixel in colored reference and target frame(s)
    for idx in [1, 3]:
        data = df.MapDataComponent(data, lambda images: [cv2.resize(image, (32, 32)) for image in images], index=idx)
        data = df.MapDataComponent(data, lambda images: [cv2.cvtColor(np.float32(image / 255.0), cv2.COLOR_BGR2Lab)[:, :, 1:] for image in images], index=idx)
        data = df.MapDataComponent(data, lambda images: [np.array([np.argmin(np.linalg.norm(centroids-v, axis=1)) for v in image.reshape((-1, 2))]).reshape((32, 32, 1)) for image in images], index=idx)

    # create combined tensor of ref and target frames for both grayscale and colorized feature maps respectively
    # generates [input, output]
    data = df.MapData(data, lambda dp: [np.stack(dp[0] + dp[2], axis=0), np.stack(dp[1] + dp[3], axis=0)])

    data = df.MapData(data, tuple)  # for tensorflow.data.dataset
    data = df.MultiProcessPrefetchData(data, nr_prefetch=256, nr_proc=num_process)
    data = df.PrefetchDataZMQ(data, nr_proc=1)
    return data

def get_input_fn(name, centroids, batch_size=32, num_ref_frames=3, num_process=16):
    """
    Iterator over datapoints in dataflow in mini-batches
    """
    _ = name
    data = dataflow(centroids, num_ref_frames=num_ref_frames, num_process=num_process, shuffle=False)
    data.reset_state()

    def input_fn():
        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                data.get_data,
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([num_ref_frames + 1, 256, 256, 1]), tf.TensorShape([num_ref_frames + 1, 32, 32, 1]))
            ).batch(batch_size)
        return dataset
    return input_fn

def main(args):
    cfg = Config(args.config) if args.config else Config()
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s', args, cfg, device_info)

    # load centroids from results of clustering
    with open(args.centroids, 'rb') as centroids_file:
        centroids = np.load(centroids_file)
    num_colors = centroids.shape[0]

    input_functions = {
        'train': get_input_fn(
            'train', centroids,
            cfg['mode']['train']['batch_size'],
            num_ref_frames=args.num_ref_frames, num_process=args.num_process
        ),
        'eval': get_input_fn(
            'test', centroids,
            cfg['mode']['eval']['batch_size'],
            num_ref_frames=args.num_ref_frames, num_process=max(1, args.num_process // 4)
        )
    }

    cfg.clear()

    # configure ResNet model
    ### understand code from here
    model_fn = model.Colorizer.get('resnet', model.ResNetColorizer, log_steps=1, num_ref_frames=args.num_ref_frames, num_colors=num_colors, predict_direction=args.direction)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=100,
        save_checkpoints_secs=None,
        save_checkpoints_steps=1000,
        save_summary_steps=10,
        session_config=None
    )
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer(
        learning_rate=args.lr
    )
    hparams = tf.contrib.training.HParams(**hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams
    )

    ### what is steps in epoch?
    for epoch in range(args.epoch):
        estimator.train(input_fn=input_functions['train'], steps=1000)
        estimator.evaluate(input_fn=input_functions['eval'], steps=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--centroids', type=str, default=DEFAULT_CENTROIDS_PATH)
    parser.add_argument('--num-ref-frames', type=int, default=3)
    parser.add_argument('--direction', type=str, default='backward', help='[forward|backward] default: backward')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--num-process', type=int, default=16)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(args)
