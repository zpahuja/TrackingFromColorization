"""
Read Kinetics video files and return iterator over frames for all videos.
See also: __iter__() docstring
"""
import os
import time
import logging
import itertools
import cv2
import numpy as np
import ujson as json


LOGGER = logging.getLogger(__name__)


class Kinetics():
    """
    Get frames from Kinetics video dataset.
    """
    def __init__(self, base_path, shuffle=False, num_frames=1, skips=(0,)):
        self.base_path = base_path
        self.shuffle = shuffle
        self.num_frames = num_frames
        self.skips = skips
        self.metas = []

        metas = json.load(open(os.path.join(base_path, 'kinetics_train.json')))
        self.keys = sorted(metas.keys())
        for _, key in enumerate(self.keys):
            metas[key]['key'] = key
            self.metas.append(metas[key])
        self.index = list(range(len(self.keys)))

    @property
    def name(self):
        return 'kinetics'

    @property
    def names(self):
        """
        Get list of video filenames
        """
        if self.shuffle:
            self.index = np.random.permutation(self.index)
        return [self.metas[idx]['key'] for idx in self.index]

    def reset_state(self):
        np.random.seed(int(time.time() * 1e7) % 2**32)

    def __len__(self):
        return len(self.index)

    def size(self, name=None):
        if name is None:
            return len(self.metas)
        return self.__len__()

    def get_filepath(self, filename):
        """
        Get filepath given filename (without .mp4 extension)
        """
        if filename not in self.keys:
            raise KeyError('File "%s.mp4" does not exist' % filename)
        filepath = os.path.join(self.base_path, 'processed', filename + '.mp4')
        exists = os.path.exists(filepath)
        return exists, filepath

    def get_data(self, num_frames=None, skips=None):
        return self.__iter__(num_frames, skips)

    def __iter__(self, num_frames=None, skips=None):
        """
        Get iterator containing frames from all video files in batches.
        :param num_frames: batch size
        :param skips: number of frames to skip
        :return: iterator of [batch size for each video, list of image frames]
        """
        num_frames = num_frames if num_frames else self.num_frames
        skips = skips if skips else self.skips

        for fname in self.names:
            exists, filepath = self.get_filepath(fname)
            if not exists:
                continue

            batch_index = -1
            images = []
            capture = cv2.VideoCapture(filepath)

            for _, num_skips in itertools.cycle(enumerate(skips)):
                # reset after yielding iterator for num_frames of current video
                if len(images) == num_frames:
                    images = []

                # skip frames
                for _ in range(num_skips):
                    capture.read()

                ret, image = capture.read()

                # terminate when end of video file reached
                if not ret:
                    break

                images.append(image)
                if len(images) == num_frames:
                    batch_index += 1
                    yield [batch_index, images]
