"""
Module containing miscellaneous utilities to log and save information. If you are looking for logging options
look for the :any:`utils.logger` submodule.
"""

import os
import shutil
from typing import Dict

import numpy as np
import tensorflow as tf
import pandas as pd
from moviepy.editor import ImageSequenceClip

import settings

from .logger import init_logger
from .results import all_results

# Only export the functions that we need
__all__ = ['init_logger', 'movie', 'save_results', 'all_results']


def movie(filename: str, array: np.ndarray, fps: int = 10, scale: float = 1.0):
    """
    Create a mp4 movie file from a 3D numpy array

    >>> import numpy as np
    >>> X = np.random.randn(100, 64, 64)
    >>> movie('test.mp4', X)

    :param filename: The filename of the gif to write to
    :param array: A numpy array that contains a sequence of images
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


def save_results(sess: tf.Session, results: Dict[str, pd.DataFrame], path: str):
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
    :param results: List with tuples with a name and a :class:`pandas.DataFrame` of results that should be saved.
                    the :class:`pandas.DataFrame` should contain at least the columns
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

    for name, result in results.items():
        merged = _select_time_age(clinical_info, result)
        merged.to_csv(os.path.join(path, f"{name}_results.csv"))


def _select_time_age(clinical_info: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merge = pd.merge(clinical_info, results, left_on='id', right_on='pA')
    merge = merge[['age', 'time', 'pA', 'pB', 'labels', 'predictions', 'probabilities', 'gather_a',
                   'gather_b']]
    merge = merge.rename(index=str, columns={'age': 'age_a', 'time': 'time_a'})

    merge = pd.merge(clinical_info, merge, left_on='id', right_on='pB')
    merge = merge[['age_a', 'age', 'time_a', 'time', 'pA', 'pB', 'labels', 'predictions', 'probabilities',
                   'gather_a', 'gather_b']]
    merge = merge.rename(index=str, columns={'age': 'age_b', 'time': 'time_b'})
    return merge


class ArgRange(float):
    def __new__(cls, start: float, end: float):
        return float.__new__(cls)

    def __init__(self, start: float, end: float):
        super().__init__()

        self.start = start
        self.end = end

    def __eq__(self, other: float) -> bool:
        return self.start <= other <= self.end

    def __contains__(self, item: float) -> bool:
        return self.__eq__(item)

    def __repr__(self) -> str:
        return f"{self.start} - {self.end}"
