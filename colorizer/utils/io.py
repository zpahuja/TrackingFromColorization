"""
Video reader and writer.

See Reader and Writer class docstrings
"""
import os
import glob
import logging
import natsort
import cv2
import imageio


LOGGER = logging.getLogger(__name__)


def reader(path):
    """Read video file agnostic to file type."""
    if os.path.isfile(path):
        return VideoReader(path)
    if os.path.isdir(path):
        return ImageReader(path)
    raise NotImplementedError


class VideoReader():
    """Iterable for video frames from mp4 file for Kinetics dataset."""

    def __init__(self, filepath):
        self.capture = cv2.VideoCapture(filepath)

    def __iter__(self):
        while True:
            ret, image = self.capture.read()
            if not ret:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield image


class ImageReader():
    """Read video frames from directory containing images for DAVIS dataset."""

    def __init__(self, dirpath):
        filenames = glob.glob(os.path.join(dirpath, '*'))
        self.filenames = natsort.natsorted(
            filenames, alg=natsort.ns.IGNORECASE)

    def __iter__(self):
        for i in range(len(self.filenames)):
            fname = self.filenames[i]
            try:
                image = cv2.imread(fname, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield image
            except Exception as e:
                LOGGER.warning('%s: %s (%s)', type(e), str(e), fname)


def writer(path):
    """Write video frames agnostic to output file format."""
    extension = os.path.splitext(path)[1]

    if extension == '.mp4':
        return VideoWriter(path)
    if extension == '.gif':
        return GifWriter(path)
    if not extension:
        return ImageWriter(path)

    LOGGER.error('Writer does not support %s file format', extension)
    raise NotImplementedError


class Writer():
    def __init__(self, path):
        self.path = path

    def write(self, images):
        pass


class ImageWriter(Writer):
    """
    Write video frames as images to directory.
    """

    def __init__(self, dirpath, extension='jpg'):
        super(ImageWriter, self).__init__(dirpath)
        self.extension = extension

    def write(self, images, images_color_space='RGB'):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for i, image in enumerate(images):
            if images_color_space == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('%s/%04d.%s' % (self.path, i, self.extension), image)


class GifWriter(Writer):
    """
    Write video as GIF file.
    """

    def write(self, images, images_color_space='RGB'):
        with imageio.get_writer(self.path, mode='I') as writer:
            for image in images:
                if images_color_space == 'BGR':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                writer.append_data(image)


class VideoWriter(Writer):
    """
    Write video as MP4 file.
    """

    def write(self, images, images_color_space='RGB', fps=24.0):
        output_format = 'MP4V'
        fourcc = cv2.VideoWriter_fourcc(*output_format)
        height, width, _ = images[0].shape

        video_writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

        for image in images:
            if images_color_space == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)

        video_writer.release()
