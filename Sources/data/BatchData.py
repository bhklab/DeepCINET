import os
from itertools import takewhile, islice, repeat
from typing import Iterator, List

import numpy as np


from .DataStructures import PairComp, PairBatch


class BatchData:
    def __init__(self):
        self._data_path = os.getenv("DATA_PROCESSED")

    def batches(self, elems: List[PairComp], batch_size: int=64) -> Iterator[PairBatch]:
        total_batches = (len(elems) // batch_size) + (1 if len(elems) % batch_size != 0 else 0)
        for i, values in enumerate(self._split(iter(elems), batch_size)):
            values = list(values)
            image_ids = set(name for p in values for name in (p.p1, p.p2))
            print("Loading batch {} of {}, {} images".format(i, total_batches, len(image_ids)))

            images = {idx: np.load(os.path.join(self._data_path, idx, idx + ".npz")).items() for idx in image_ids}
            yield PairBatch(pairs=values, images=images)

    @staticmethod
    def _split(it: Iterator, n: int) -> Iterator[Iterator]:
        return takewhile(bool, (list(islice(it, n)) for _ in repeat(None)))

