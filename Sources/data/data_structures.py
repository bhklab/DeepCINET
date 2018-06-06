from typing import NamedTuple, Dict

import pandas as pd


class PseudoDir(NamedTuple):
    """
    Represent a pseudo directory, useful to store the information from a :class:`os.DirEntry`
    """
    name: str
    path: str
    is_dir: bool


class PairBatch(NamedTuple):
    """
    Tuple class for a batch of data that can be used to train a machine learning model. It's prepared to load as
    minimum as possible images since the images can be gathered to create the pairs.

    >>> import numpy as np
    >>> batch = PairBatch(...)
    >>> images_a = np.take(batch.patients["images"], batch.pairs["pA_id"])
    >>> images_b = np.take(batch.patients["images"], batch.pairs["pB_id"])
    >>> len(images_a) == len(images_b)  # Must-have condition
    True

    :ivar PairBatch.pairs: :class:`pandas.DataFrame` containing the information regarding the batch pairs. It has
                           the following columns:

                             - ``pA``: String key for the pair's element A
                             - ``pB``: String key for the pair's element B
                             - ``pA_id``: Index for the pair's element A
                             - ``pB_id``: Index for the pair's element B
                             - ``comp``: :any:`bool` that it's true if :math:`T(p_a) < T(p_b)`, where :math:`T(x)`
                               is the patient's survival time
                             - ``label``: Same as ``comp`` but in float format where ``1.`` stands for :any:`True`
                             - ``distance``: Float with the distance in time between the two patient's normalized

    :ivar PairBatch.patients: :class:`pandas.DataFrame` with the patient's information. It has the following columns:

                                - ``ids``: Id identifying each one of the images and features
                                - ``images``: :class:`numpy.ndarray` with shape ``[64, 64, 64, 1]`` containing the
                                  patient's scan
                                - ``features``: :class:`numpy.ndarray` with shape ``[NUMBER_FEATURES, 1]``

    :vartype PairBatch.pairs: pandas.DataFrame
    :vartype PairBatch.ids_inverse: dict[int, str]
    """
    pairs: pd.DataFrame
    patients: pd.DataFrame
    ids_map: Dict[str, int]
