import os

import numpy as np
import tensorflow as tf
import pandas as pd
from moviepy.editor import ImageSequenceClip

import settings
from data.data_structures import PairBatch

from .logger import get_logger, init_logger

# Only export the functions that we need
__all__ = ['get_logger', 'init_logger', 'movie', 'save_model']


def movie(filename: str, array: np.ndarray, fps: int = 10, scale: float = 1.0):
    """
    Create a mp4 movie file from a 3D numpy array

    >>> import numpy as np
    >>> X = np.random.randn(100, 64, 64)
    >>> movie('test.mp4', X)

    :param filename: The filename of the gif to write to
    :param array: A numpy array that contains a squence of images
    :param fps: frames per second (default: 10)
    :param scale: how much to rescale each image by (default: 1.0)

    Inspired in https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59

    """

    # ensure that the file has the .gif extension
    basename, _ = os.path.splitext(filename)
    filename = basename + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis]*np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_videofile(filename, fps=fps)
    return clip


def save_model(sess: tf.Session, train_pairs: pd.DataFrame, test_pairs: pd.DataFrame, directory: str):
    saver = tf.train.Saver()
    saver.save(sess, settings.SESSION_SAVE_PATH)
    pass
