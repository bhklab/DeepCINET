"""
Module containing miscellaneous utilities to log and save information. If you are looking for logging options
look for the :any:`utils.logger` submodule.
"""

import os

import numpy as np
from moviepy.editor import ImageSequenceClip

from .logger import init_logger
from .results import all_results, save_results

# Only export the functions that we need
__all__ = ['init_logger', 'movie', 'save_results', 'all_results', 'ArgRange']


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


class ArgRange(float):
    """
    Class to add a range of float values as a choice when creating arguments with :any:`argparse.ArgumentParser`
    """
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

    def __str__(self) -> str:
        return self.__repr__()
