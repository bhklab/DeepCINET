"""
Module containing miscellaneous utilities to log and save information. If you are looking for logging options
look for the :any:`utils.logger` submodule.
"""

import os
import shutil

import numpy as np
import tensorflow as tf
import pandas as pd
from moviepy.editor import ImageSequenceClip

import settings

from .logger import get_logger, init_logger

# Only export the functions that we need
__all__ = ['get_logger', 'init_logger', 'movie', 'save_results']


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


def save_results(sess: tf.Session, train_results: pd.DataFrame, test_results: pd.DataFrame, path: str):
    """
    Save the current results to disk. It creates a CSV file with the pairs and its values. Keeping in
    mind that the results are pairs it uses the suffixes ``_a`` and ``_b`` to denote each member of the pair

        - ``age_a``: Age of pair's member A
        - ``age_b``: Age of pair's member B
        - ``time_a``: Survival time of pair's member A
        - ``time_b``: Survival time of pair's member B
        - ``pairs_a``: Key of pair's member A
        - ``pairs_b``: Key of pair's member B
        - ``labels``: Labels that are true if :math:`T(p_a) < T(p_b)`
        - ``predictions``: Predictions made by the current model

    Moreover, the model is also saved into disk. It can be found in the ``path/weights/`` directory and can
    loaded with Tensorflow using the following commands:

    >>> import tensorflow as tf
    >>> saver = tf.train.Saver()
    >>> with tf.Session() as sess:
    >>>     saver.restore(sess, "<path>/weights/weights.ckpt")

    :param sess: Current session that should be saved when saving the model
    :param train_results: :class:`pandas.DataFrame` containing the train results, it must have at least the columns
                          ``pairs_a``, ``pairs_b``, ``labels`` and ``predictions``.
    :param test_results: :class:`pandas.DataFrame` containing the test results, it must have at least the columns
                          ``pairs_a``, ``pairs_b``, ``labels`` and ``predictions``.
    :param path: Directory path where all the results should be saved
    """
    weights_dir = os.path.join(path, 'weights')

    # Always overwrite the previous weights
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(weights_dir)

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(weights_dir, 'weights.ckpt'))

    # Load clinical info
    clinical_info = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED, index_col=0)
    merge_train = _select_time_age(clinical_info, train_results)
    merge_test = _select_time_age(clinical_info, test_results)

    merge_train.to_csv(os.path.join(path, 'train_results.csv'))
    merge_test.to_csv(os.path.join(path, 'test_results.csv'))


def _select_time_age(clinical_info: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merge = pd.merge(clinical_info, results, left_on='id', right_on='pairs_a')
    merge = merge[['age', 'time', 'pairs_a', 'pairs_b', 'labels', 'predictions', 'probabilities', 'gather_a',
                   'gather_b']]
    merge = merge.rename(index=str, columns={'age': 'age_a', 'time': 'time_a'})

    merge = pd.merge(clinical_info, merge, left_on='id', right_on='pairs_b')
    merge = merge[['age_a', 'age', 'time_a', 'time', 'pairs_a', 'pairs_b', 'labels', 'predictions', 'probabilities',
                   'gather_a', 'gather_b']]
    merge = merge.rename(index=str, columns={'age': 'age_b', 'time': 'time_b'})
    return merge
