"""
The data package contains different classes related in how to read and pre-process data.
There are also some classes useful for getting all the different pairs in the dataset and
get it in form of different batches

"""
from data.image_data import RawData, PreProcessedData
from data.pair_data import SplitPairs, BatchData
from data.data_structures import PseudoDir, PairBatch, PairComp
