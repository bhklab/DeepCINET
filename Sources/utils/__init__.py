import os

import numpy as np
from moviepy.editor import ImageSequenceClip

from .logger import get_logger, init_logger


def movie(filename, array, fps=10, scale=1.0):
    """
    Notes
    -----

    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e

    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> movie('test.gif', X)

    :param filename: The filename of the gif to write to
    :param array: A numpy array that contains a squence of images
    :param fps: frames per second (default: 10)
    :param scale: how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_videofile(filename, fps=fps)
    return clip
