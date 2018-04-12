from typing import NamedTuple, List, Dict, Iterable


import numpy as np


class PseudoDir(NamedTuple):
    name: str
    path: str
    is_dir: bool


class PairComp(NamedTuple):
    """
    Class to compare pairs containing two ids and a comparison value saying if
    T(p1) < T(p2)
    """
    p1: str
    p2: str
    comp: bool


class PairBatch(NamedTuple):
    pairs: Iterable[PairComp]
    images: Dict[str, List[np.ndarray]]
