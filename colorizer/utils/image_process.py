"""
Utility class to resize and crop images
"""
from __future__ import absolute_import

import logging
import copy as cp
import cv2


LOGGER = logging.getLogger(__name__)


class ImageProcessor():
    @staticmethod
    def resize(small_axis=256, copy=False):
        def _resize(images):
            images = cp.deepcopy(images) if copy else images

            for idx, image in enumerate(images):
                height, width = image.shape[:2]
                aspect_ratio = 1.0 * width / height
                width = int(small_axis if aspect_ratio <= 1.0 else (small_axis * aspect_ratio))
                height = int(small_axis if aspect_ratio >= 1.0 else (small_axis / aspect_ratio))
                images[idx] = cv2.resize(image, (width, height))

            return images
        return _resize

    @staticmethod
    def crop(shape, copy=False):
        def _crop(images):
            images = cp.deepcopy(images) if copy else images
            target_height, target_width = shape[:2]

            for idx, image in enumerate(images):
                height, width = image.shape[:2]
                start_x = max(0, (width - target_width) // 2)
                start_y = max(0, (height - target_height) // 2)
                end_x, end_y = start_x + target_width, start_y + target_height
                image = image.reshape((height, width, -1))[start_y:end_y, start_x:end_x, :]
                images[idx] = image

            return images
        return _crop
