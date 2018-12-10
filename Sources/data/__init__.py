"""
The data package contains different classes related in how to read and pre-process data.

    - The :class:`~data.image_data.RawData` class represents the original raw data, it has some methods to save
      these data as a faster ``.npz`` file.
    - The :class:`~data.image_data.PreProcessedData` class represents the preprocessed data. It can generate
      these data from the raw data.
    - The :class:`~data.pair_data.SplitPairs` class helps in obtaining the different pairs that can be used when
      training the model.
    - The :class:`~data.pair_data.BatchData` class can be used to load all the information and generate batches
      that can be used to train the data step by step.

Some data structures are defined to use when running the scripts

    - The :class:`~data.data_structures.PseudoDir` class is used only to represent a directory structure when
      pre-processing the data.
    - The :class:`~data.data_structures.PairBatch` class is used to represent a batch of data, it contains all
      the elements necessary to do one batch step when training.

Relations between different modules and classes:

.. graphviz::

   digraph {
       train, preprocess, PreProcessedData, RawData, PseudoDir, SplitPairs, BatchData, PairBatch;

       subgraph R1 {
           edge [dir=none]
           PreProcessedData -> RawData;
           RawData -> PseudoDir;
           BatchData -> PairBatch;
           train -> BatchData;
           train -> SplitPairs;
           train -> PairBatch;

           preprocess -> PreProcessedData;
       }
   }

"""
from data.image_data import RawData, PreProcessedData
from data.pair_data import SplitPairs, BatchData
from data.data_structures import PseudoDir, PairBatch
from data.mrmrpy import mrmr_selection, select_mrmr_features