from typing import NamedTuple, List, Dict, Iterable, Union

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
    p1: str
    p2: str
    comp: bool


class PairBatch(NamedTuple):
    p1: List[int]
    p2: List[int]
    labels: List[int]
    # TODO: Fix the status with the images
    images: Dict[int, List[np.ndarray]]
    ids_map: Dict[str, int]
