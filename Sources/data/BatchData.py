import os
import math
from itertools import takewhile, islice, repeat
from typing import Iterator, Iterable, Generator, List, Set

import numpy as np

from .DataStructures import PairComp, PairBatch


class BatchData:
    def __init__(self):
        self._data_path = os.getenv("DATA_PROCESSED")

    def batches(self, pairs: List[PairComp], batch_size: int = 64, group_by: str = 'ids') -> Generator[PairBatch]:
        if group_by == 'ids':
            return self._batch_by_ids(pairs, batch_size)
        else:
            return self._batch_by_pairs(pairs, batch_size)

    def _batch_by_ids(self, pairs: Iterable[PairComp], batch_size: int):
        total_pairs = set(pairs)

        # Extract bath_size ids
        while len(total_pairs) > 0:
            ids = set()
            batch_pairs = set()

            # Create a batch of batch_size ids
            while len(ids) < batch_size and len(total_pairs) > 0:
                pair = total_pairs.pop()
                ids |= {pair.p1, pair.p2}
                batch_pairs.add(pair)

            # Get all the pairs that can be formed with those ids and then remove the batch pairs from the total pairs
            batch_pairs |= {x for x in total_pairs if x.p1 in ids and x.p2 in ids}
            total_pairs -= batch_pairs
            assert len(batch_pairs)*2 >= len(ids)

            yield self._create_pair_batch(batch_pairs, ids)

    def _batch_by_pairs(self, pairs: List[PairComp], batch_size: int) -> Generator[PairBatch]:
        for i, values in enumerate(self._split(pairs, batch_size)):
            print(type(values))
            values = list(values)
            yield self._create_pair_batch(values, {idx for p in values for idx in (p.p1, p.p2)})

    def _create_pair_batch(self, pairs: Iterable[PairComp], ids: Set[str]) -> PairBatch:
        """
        Given all the ids and the pairs load the npz file for all the ids and create a PairBatch with the loaded
        npz files and the pairs
        :param pairs: Pairs to be added to the PairBatch
        :param ids: npz files' ids that will be added to the PairBatch
        :return: PairBatch containing the pairs and the requested npz files loaded
        """
        images = {idx: np.load(os.path.join(self._data_path, idx, idx + ".npz")).items() for idx in ids}
        return PairBatch(pairs=pairs, images=images)

    @staticmethod
    def _split(it: Iterable, n: int) -> Iterable[Iterable]:
        """
        Given an iterable create batches of size n
        :param it: Iterable
        :param n: Batch size
        :return: Batches of size n
        """
        return takewhile(bool, (list(islice(iter(it), n)) for _ in repeat(None)))
