
import pandas as pd


class ClinicalReader:
    """ Base class which provides several utilities working with clinical information

    """
    _ID_COL = 'Study ID'
    _S_TIME = 'survival_time'
    _DROP_C = ['split']
    _DEATH = 'death'

    def __init__(self, hparams, idxs=None):
        self._clinical_csv = pd.read_csv(hparams.clinical_path) \
            .drop(columns=self._DROP_C)
        if idxs is not None:
            self._clinical_csv = self._clinical_csv.iloc[idxs]
        self._id_list = [x for x in self._clinical_csv[self._ID_COL].values]
        self._index_map = dict(zip(
            self._clinical_csv[self._ID_COL].values,
            list(range(len(self._clinical_csv[self._ID_COL].values)))
            ))
        self._event_list = [x for x in self._clinical_csv[self._DEATH]]

    def get_id_from_index(self, idx):
        return self._id_list[idx]

    def get_index_from_id(self, p_id):
        return self._index_map[p_id]

    def get_survival_time_from_index(self, idx):
        return self._clinical_csv[self._S_TIME].iloc[idx]

    def get_event_from_index(self, idx):
        return self._clinical_csv[self._DEATH].iloc[idx]



