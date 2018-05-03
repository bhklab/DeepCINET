from typing import NamedTuple, List, Dict

import numpy as np


class PseudoDir(NamedTuple):
    """
    Represent a pseudo directory, useful to store the information from a :class:`os.DirEntry`
    """
    name: str
    path: str
    is_dir: bool


class PairComp(NamedTuple):
    """
    Tuple class to compare pairs containing two ids and a comparison value saying if
    :math:`T(p_a) < T(p_b)`

    :ivar PairComp.comp: True if :math:`T(p_a) < T(p_b)`
    :ivar PairComp.p_a: Key for pair's element A
    :ivar PairComp.p_b: Key for pair's element B
    :vartype PairComp.comp: bool
    :vartype PairComp.p_a: str
    :vartype PairComp.p_b: str
    """
    p_a: str
    p_b: str
    comp: bool


class PairBatch(NamedTuple):
    """
    Tuple class for a batch of data that can be used to train a machine learning model. It's prepared to load as
    minimum as possible images since the images can be gathered to create the pairs.

    >>> import numpy as np
    >>> batch = PairBatch(...)
    >>> images_a = np.take(batch.images, batch.pairs_a)
    >>> images_b = np.take(batch.images, batch.pairs_b)
    >>> len(images_a) == len(images_b) == len(images_b)  # Must-have condition
    True
    >>>

    :ivar PairBatch.pairs_a: List of indices where the images can be selected to get the pair's element A
    :ivar PairBatch.pairs_b: List of indices where the images can be selected to get the pair's element B
    :ivar PairBatch.labels: Labels for the prediction, the possible values for each label are ``0.0`` or ``1.0``
    :ivar PairBatch.images: All the images for the indices, note that the indices for the pairs must be between
                            ``0`` and ``len(batch.images) - 1``
    :ivar PairBatch.features: Radiomic features
    :ivar PairBatch.ids_map: Dictionary that map each key as :any:`string` with its corresponding index as an
                             :any:`int`
    :ivar PairBatch.ids_inverse: Inverse mapping for the dictionary that goes from index as an :any:`int` to
                                 its corresponding key as :any:`string`
    :vartype PairBatch.pairs_a: list[int]
    :vartype PairBatch.pairs_b: list[int]
    :vartype PairBatch.pairs_tag: list[str]
    :vartype PairBatch.labels: list[float]
    :vartype PairBatch.images: numpy.ndarray
    :vartype PairBatch.features: numpy.ndarray
    :vartype PairBatch.ids_map: dict[str, int]
    :vartype PairBatch.ids_inverse: dict[int, str]
    """
    pairs_a: List[int]
    pairs_b: List[int]
    labels: List[float]
    # TODO: Fix the status with the images
    images: np.ndarray
    features: np.ndarray
    ids_map: Dict[str, int]
    ids_inverse: Dict[int, str]
