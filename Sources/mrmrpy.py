#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

#os.environ["RHOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

'''
print(robjects.r("version"))
base = importr('base')
print(base._libPaths())
print(robjects.r('R.home()'))
print(base._libPaths())
'''


class mrmrpy:

    def __init__(self):

        # Import the necessary r libraries needed for mRMRe
        self.matrix = rpackages.importr('Matrix')
        self.survival = rpackages.importr('survival')
        self.igraph = rpackages.importr('igraph')
        self.mrmre = rpackages.importr('mRMRe')

        # Launch the mRMRe
        robjects.r('ShowClass("mRMRe.Filter")')
        robjects.r('set.thread.count(2)')

    def mrmr_data(self,
                  features : pd.DataFrame = None,
                  clinical_info : pd.DataFrame = None):
        '''
        This function "merges" the features dataset and clinical info, to create a new features dataframe which meets
        the criteria of mRMRe package, and convert the dataframe into mRMRe.data object

        :param features      : The radiomics features or clinical info dataset
        :param clinical_info : The clinical info dataset (only the 'time' field is needed in this function)
        :return: The mRMR.data object of new feature dataframe
        '''

        # Merge the features and labels (the survival time, which is the 'time' field of clinical information)
        features = features.T
        samples = clinical_info['id']
        features = features.loc[samples]

        # Concatenate the survival time to the features dataset
        features = features.reset_index(drop=True)
        clinical_info = clinical_info.reset_index(drop=True)
        features = features.join(pd.DataFrame(clinical_info['time']))

        # Make the survial time (label) as the first column in the dataframe
        cols = features.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        features = features[cols]

        # Convert the features dataframe into the mRMRe.data object

        return self.mrmre.mRMR.data(data=features)

    def mrmr_ensemble(self,
                      data,
                      solution_count : int = 1,
                      feature_count : int = 1,
                      method : str = None):
        '''
        This function call the ensemble method of mRMR, and return the selected features
        :param data          : The input mRMRe.Data object (features)
        :param solution_count: The number of solutions wanted
        :param feature_count : The number of target features wanted in one solution
        :param method        : The mRMR ensemble method, could be "bootstrap" and "exhaustive"
        :return: The index list of features selected
        '''

        feature_selected = self.mrmre.mRMR.ensemble(solution_count=solution_count,
                                                    feature_count=feature_count,
                                                    data=data,
                                                    target_indices=1,
                                                    method=method)

        return self.mrmre.solutions(feature_selected)[0]

