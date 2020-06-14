from data.clinical_reader import ClinicalReader


class ClinicalLoader(ClinicalReader):
    """Class in charge of loading clinical variables

    Provides functions to build all the concordant pairs as well as
    utilities to perform one hot encoding
    """
    def __init__(self, hparams, idxs):
        ClinicalReader.__init__(self, hparams, idxs)

    def get_concordant_pair_list(self, num_neighbours):
        """Return the list of the k closest concordant pairs for each patient

        For each patient, it finds the k closest concordant
        patients which die before and after.

        :param int num_neighbours: Number of pairs per patient to consider
        :return: List containing tuples of the concordant pairs
        """
        uncen_list = []
        pair_list = []
        for i in range(len(self._id_list)):
            idxi = i
            for j in range(len(uncen_list)):
                idxj = uncen_list[-(j+1)]
                pair_list.append({
                    'idxA': idxi,
                    'idxB': idxj,
                    'label': self.get_relationship_from_index(idxi, idxj)}
                )
                if num_neighbours == j:
                    break

            if self.get_event_from_index(idxi):
                uncen_list.append(idxi)

        return pair_list

    def get_patient_list(self):
        return self._id_list

    def get_relationship_from_index(self, idx1, idx2):
        return int(self.get_survival_time_from_index(idx1)
                   > self.get_survival_time_from_index(idx2))
