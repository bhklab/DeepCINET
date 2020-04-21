from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd


class CoxModel():
    def __init__(self, hparams):
        self.hparams = hparams
        self.clinical_csv = pd.read_csv(hparams.clinical_path, index_col=0)
        self.radiomics_path = hparams.radiomics_path
        cols = ['lbp-2D_glszm_SmallAreaEmphasis',
                'lbp-2D_glszm_SmallAreaHighGrayLevelEmphasis',
                'lbp-2D_glszm_SmallAreaLowGrayLevelEmphasis',
                'lbp-3D-m1_glszm_SmallAreaEmphasis',
                'lbp-3D-m1_glszm_SmallAreaHighGrayLevelEmphasis',
                'lbp-3D-m1_glszm_SmallAreaLowGrayLevelEmphasis',
                'lbp-3D-m2_glszm_SmallAreaEmphasis',
                'lbp-3D-m2_glszm_SmallAreaHighGrayLevelEmphasis',
                'lbp-3D-m2_glszm_SmallAreaLowGrayLevelEmphasis']

        low_var_col = ['lbp-2D_glcm_DifferenceEntropy',
                       'lbp-2D_glcm_JointEntropy',
                       'lbp-2D_glcm_SumEntropy',
                       'lbp-3D-m1_glcm_DifferenceEntropy',
                       'lbp-3D-m1_glcm_JointEntropy',
                       'lbp-3D-m1_glcm_SumEntropy',
                       'lbp-3D-m2_glcm_DifferenceEntropy',
                       'lbp-3D-m2_glcm_JointEntropy',
                       'lbp-3D-m2_glcm_SumEntropy',
                       'wavelet-HHH_glcm_ClusterProminence']

        clinical_var = ['id', 'event', 'time']
        if(hparams.use_clinical):
            clinical_var.extend([
                'Age',
                'Sex',
                'ECOG PS',
                'Smoking Hx',
                'Drinking hx',
                'T',
                'N',
                'Stage'
            ])

        if(self.hparams.use_radiomics):
            self.radiomics_csv = pd.read_csv(self.radiomics_path, index_col=0)
            self.radiomics_csv.drop(columns = cols, inplace= True)
            self.radiomics_csv.drop(columns = low_var_col, inplace= True)

        self.merge_csv = pd.merge(self.radiomics_csv, self.clinical_csv[clinical_var].fillna(0), on='id')
        self.merge_csv.drop(columns = 'id', inplace=True)

    def fit(self, train_ids, val_ids):
        cph = CoxPHFitter(penalizer = 0.1)
        cph.fit(self.merge_csv.iloc[train_ids], duration_col='time', event_col='event', show_progress=True, step_size=0.2)
        # cph.print_summary()
        val_csv = self.merge_csv.iloc[val_ids]
        CI = concordance_index(val_csv['time'], -cph.predict_partial_hazard(val_csv).values, val_csv['event'])
        return CI


    def del_low_var(self):
        low_variance = self.merge_csv.columns[df.var(0) < 1e-3]
        self.merge_csv.drop(low_variance, axis=1, inplace=True)

    def volume_only(self, train_ids):
        val_csv = self.merge_csv.iloc[train_ids]
        print(concordance_index(val_csv['time'], -val_csv['original_shape_VoxelVolume'], val_csv['event']))



