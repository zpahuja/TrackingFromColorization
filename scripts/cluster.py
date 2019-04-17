"""
Cluster color dimensions in LAB colorspace for Kinetics & CIFAR10 datasets
"""
import sys
import cv2
import argparse
import logging
import numpy as np
import tensorpack.dataflow as df
from sklearn.cluster import KMeans

sys.path.append('.')
sys.path.append('/home/nfs/zpahuja2/tracking_from_colorization')  # vision clus

from colorizer.dataset import Kinetics
from colorizer.config import Config


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='centroids.npy')
    parser.add_argument('-k', '--num-clusters', type=int, default=16)
    parser.add_argument('-n', '--num-samples', type=int, default=50000)
    parser.add_argument('--name', type=str, default='kinetics')
    parser.add_argument('-l', '--log', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # func with o k n name log(default)
    # func to get data flow
    # func for mp4 / gif?
    # func to get frame for one video?
    # func to get clustered results?

    # function to call script, log level default warning for function, info for script

    cfg = Config()
    kinetics_dirpath = cfg['data_dir']['kinetics']

    # configure logger
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log)
    logging.info('args: %s', args)

    if args.name == 'kinetics':
        # get every frame of
        ds = Kinetics(kinetics_dirpath, num_frames=1, skips=[0], shuffle=False)

        # keep only first frame of each sub-video
        # [sub_video_idx, frames[]] -> [first_frame, sub_video_idx]
        ds = df.MapData(ds, lambda dp: [dp[1][0], dp[0]])
    else:
        ds = df.dataset.Cifar10('train', shuffle=False)

    logging.info('Downsampling frames to 32x32 resolution')
    ds = df.MapDataComponent(ds, lambda image: cv2.resize(image, (32, 32)))
    logging.info('Converting RGB to Lab color space')
    ds = df.MapDataComponent(ds, lambda image: cv2.cvtColor(np.float32(image / 255.0), cv2.COLOR_RGB2Lab))
    ds = df.MapDataComponent(ds, lambda image: image[:, :, 1:])
    ds = df.MapDataComponent(ds, lambda image: image.reshape((-1, 2)))
    ds = df.RepeatedData(ds, -1)
    ds.reset_state()

    generator = ds.get_data()

    samples = []
    for _ in range(args.num_samples):
        samples.append(next(generator)[0])
    vectors = np.array(samples).reshape((-1, 2))
    logging.info('Vectorized images in the shape: %s', vectors.shape)

    kmeans = KMeans(args.num_clusters).fit(vectors)
    logging.info('Fitted kmeans clustering')

    centroids = np.array(kmeans.cluster_centers_)

    with open(args.output, 'wb') as f:
        centroids.dump(f)
    logging.info('Finished writing centroids to "%s"', args.output)
