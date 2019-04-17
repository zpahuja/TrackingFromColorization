"""
Read DAVIS 2017 video frames and annotations,
and return iterator over frames and annotations for all videos.
See also: __iter__() docstring
See also: Kinetics() which has similar API, without annotated frames
"""
import os
import time
import logging
import numpy as np
import cv2


LOGGER = logging.getLogger(__name__)


class Davis():
    """
    Get frames and annotations from DAVIS video object segmentation dataset.
    """

    def __init__(self, base_path, set='train', resolution='1080p', shuffle=False, num_frames=1):
        self.base_path = base_path
        self.shuffle = shuffle
        self.num_frames = num_frames
        self.annotation_dir = os.path.join(
            base_path, 'Annotations', resolution)
        self.image_dir = os.path.join(base_path, 'JPEGImages', resolution)
        imageset_path = os.path.join(
            base_path, 'ImageSets', resolution, set + '.txt')
        self._names = [line.split('/')[-2]
                       for line in open(imageset_path).readlines()]
        self.index = list(range(len(self._names)))

    @property
    def name(self):
        return 'davis'

    @property
    def names(self):
        if self.shuffle:
            self.index = np.random.permutation(self.index)
        return [self._names[idx] for idx in self.index]

    def reset_state(self):
        np.random.seed(int(time.time() * 1e7) % 2**32)

    def __len__(self):
        return len(self.index)

    def size(self):
        return self.__len__()

    def get_data(self, num_frames=None):
        return self.__iter__(num_frames)

    def __iter__(self, num_frames=None):
        """
        Get iterator containing frames and annotations from all video files.
        :param num_frames: batch size
        :return: iterator of [batch size per video, list of image frames, list of annotated frames]
        """
        num_frames = num_frames if num_frames else self.num_frames

        for dirname in self.names:
            batch_index = -1
            images, annotations = [], []

            for image, annotation in zip(self.get_image_filepaths(dirname), self.get_annotation_filepaths(dirname)):
                # reset after yielding iterator for num_frames of current video
                if len(images) == num_frames:
                    images, annotations = [], []

                images.append(cv2.imread(image))
                annotations.append(cv2.imread(annotation))

                if len(images) == num_frames:
                    batch_index += 1
                    yield [batch_index, images, annotations]

    def get_frame_filepaths(self, dirpath):
        """
        Get list of filepaths of all frames in video directory in sorted order
        """
        fnames = os.listdir(dirpath)
        fnames.sort()
        return [os.path.join(dirpath, fname) for fname in fnames]

    def get_image_filepaths(self, dirname):
        """
        Get filepaths of all image frames for video directory in sorted order
        """
        image_dirpath = os.path.join(self.image_dir, dirname)
        return self.get_frame_filepaths(image_dirpath)

    def get_annotation_filepaths(self, dirname):
        """
        Get filepaths of all annotated frames for video directory in sorted order
        """
        annotation_dirpath = os.path.join(self.annotation_dir, dirname)
        return self.get_frame_filepaths(annotation_dirpath)
