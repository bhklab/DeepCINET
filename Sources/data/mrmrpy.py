#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

#os.environ["RHOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

import logging
matrix = importr('Matrix')
survival = importr('survival')
igraph = importr('igraph')
mrmre = importr('mRMRe')

def mrmr_selection(features : pd.DataFrame,
                    clinical_info : pd.DataFrame,
                    solution_count: int,
                   feature_count: int):
    '''
     This function use mrmr feature selection to select specific number of features

     :param features      : The radiomics features or clinical info dataset
     :param clinical_info : The clinical info dataset (only the 'time' and 'event' field is needed in this function)
     :param solution_count: The number of solution which is used in ensamble function [Currently we only use 1 solution]
     :param feature_count : The number of features which should be chosen
     :return: New feature dataframe
     '''


    #Call the r wrapper to use mrmre for selecting features
    #This function use mRMR.ensemble to return max relevant min redundant features
    robjects.r('''
                # create a function `mrmrSelection`
                mrmrSelection <- function(df_all, feature_count= feature_count, solution_count = solution_count) {
                    df_all$event <- as.numeric(df_all$event)
                    df_all$surv <- with(df_all, Surv(time, event))
                    df_all <- subset(df_all, select = -c(time,event) )
                    df_all <- df_all[,colSums(is.na(df_all))<nrow(df_all)]
                    surv_index = grep("surv", colnames(df_all)) 
                    feature_data <- mRMR.data(df_all)
                    feature_selected <- mRMR.ensemble(data = feature_data, target_indices = surv_index,feature_count= feature_count, solution_count = solution_count)
                    result <-  as.data.frame(solutions(feature_selected)[1])

                }
                ''')
    r_mrmrSelection = robjects.r['mrmrSelection']
    # Merge the features and labels (the survival time, which are the 'time' and 'event'  field of clinical information)
    features = features.T
    samples = clinical_info['id']
    features = features.loc[samples]
    clinical_info.set_index('id', inplace=True)
    features[['time', 'event']] = clinical_info[['time', 'event']]
    all_features = r_mrmrSelection(features, feature_count= feature_count, solution_count = solution_count)
    selected_features = pandas2ri.ri2py_dataframe(all_features)
    #the dataframe index is diffrent in R and python
    selected_features = selected_features.sub(1)
    return selected_features.iloc[:,0]

