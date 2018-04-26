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
    Class to compare pairs containing two ids and a comparison value saying if
    T(p1) < T(p2)

    :attr comp: True if T(p1) < T(p2)
    """
    p_a: str
    p_b: str
    comp: bool


class PairBatch(NamedTuple):
    pairs_a: List[int]
    pairs_b: List[int]
    labels: List[int]
    # TODO: Fix the status with the images
    images: np.ndarray
    ids_map: Dict[str, int]
