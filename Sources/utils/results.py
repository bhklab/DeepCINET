"""
Module to work with result data
"""
import glob
import collections
import logging
import os
import shutil
from typing import Dict

import pandas as pd
import numpy as np
import tensorflow as tf

import config as settings

target_df = None


def save_ml_results(results: Dict[str, pd.DataFrame], path: str):
    """
    Save the cox results to disk. It creates a CSV file with the pairs and its values. Keeping in
    mind that the results are pairs it uses the suffixes ``_a`` and ``_b`` to denote each member of the pair

    """

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for name, result in results.items():
        result = result[['pA', 'pB', 'distance', 'predict_a', 'predict_b', 'target_a', 'target_b', 'comp', 'predict_comp']]
        result.to_csv(os.path.join(path, f"{name}_results.csv"))


# Get the mixed results
def all_results(path, select_type, predictions=False, elem_folds=False):
    """
    Get all the results when using multiple folds in training

    :param path: Path where the results should be located. Folders starting with ``fold*`` will be searched
                 accordingly for results.
    :param select_type: Type of results to be selected, it can be ``train``, ``test`` or ``mixed``.
    :param predictions: Whether to load all the predictions or not to create a data frame with all the predictions.
                        Useful to create ROC curves.
    :param elem_folds: Whether to load or not the predictions and store them in a dictionary where the key is the
                       most repeated key and the value is the pandas :class:`pandas.DataFrame` with the comparisons
    :return:
    """
    global target_df

    if target_df is None:
        target_df = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED)
    logger = logging.getLogger(__name__)

    logger.debug(f"Searching on {path} {select_type}")
    files = glob.glob(path + f"/fold*/{select_type}*.csv")
    logger.debug(f"Found {len(files)}")

    df_list = []
    elem_comparisons = {}
    unique_ids = target_df['id'].unique()
    for file in files:
        df = pd.read_csv(file, index_col=0)

        elem_right = len(df[df["labels"].astype(bool) == df["predictions"].astype(bool)])
        elem_count = len(df)

        ids = np.concatenate((df['pA'].values, df['pB'].values))
        ids = collections.Counter(ids)
        key, count = ids.most_common(1)[0]

        # No LOOCV
        if len(files) < len(target_df):
            gather = np.array([0, 0])
            is_censored = False
            time = 0
        else:
            if key not in unique_ids:
                continue
            gather = df.loc[df['pA'] == key, 'gather_a'].values
            gather = np.append(gather, df.loc[df['pB'] == key, 'gather_b'].values)

            is_censored = not target_df.loc[target_df['id'] == key, 'event'].values[0]
            time = target_df.loc[target_df['id'] == key, 'time'],

        df_list.append(pd.DataFrame({
            "id": [key],
            "right": [elem_right],
            "total": [elem_count],
            "censored": [is_censored],
            "time": [time],
            "file": [file],
            "gather": [gather.mean()]
        }))

        if elem_folds or predictions:
            elem_comparisons[key] = df

    if predictions:
        predictions_df: pd.DataFrame = pd.concat(elem_comparisons.values(), ignore_index=True)
    else:
        predictions_df = None

    results_df: pd.DataFrame = pd.concat(df_list, ignore_index=True)
    if select_type == "mixed":
        results_df['c-index'] = results_df['right'] / results_df['total']

    no_cens_results = results_df.loc[~results_df['censored']]

    logger.info(f"Finished {path} {select_type}\n")
    return (results_df, no_cens_results), predictions_df, elem_comparisons


def _select_time(target_data_frame: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    merge = pd.merge(target_data_frame, results_df, left_on='id', right_on='pA')
    merge = merge[['time', 'pA', 'pB', 'labels', 'predictions', 'probabilities', 'gather_a',
                   'gather_b']]
    merge = merge.rename(index=str, columns={'time': 'time_a'})

    merge = pd.merge(target_data_frame, merge, left_on='id', right_on='pB')
    merge = merge[['time_a', 'time', 'pA', 'pB', 'labels', 'predictions', 'probabilities',
                   'gather_a', 'gather_b']]
    merge = merge.rename(index=str, columns={'time': 'time_b',
                                             'gather_a': 'predict_time_a',
                                             'gather_b': 'predict_time_b'})
    merge['real_dist'] = merge['time_b'] - merge['time_a']
    return merge


def _select_target(target_data_frame: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    merge = pd.merge(target_data_frame, results_df, left_on='id', right_on='pA')
    merge = merge[['target', 'pA', 'pB', 'labels', 'predictions', 'probabilities', 'gather_a',
                   'gather_b']]
    merge = merge.rename(index=str, columns={'target': 'target_a'})

    merge = pd.merge(target_data_frame, merge, left_on='id', right_on='pB')
    merge = merge[['target_a', 'target', 'pA', 'pB', 'labels', 'predictions', 'probabilities',
                   'gather_a', 'gather_b']]
    merge = merge.rename(index=str, columns={'target': 'target_b',
                                             'gather_a':'predict_a',
                                             'gather_b':'predict_b'})
    merge['real_dist'] = merge['target_b'] - merge['target_a']
    return merge


def df_results(target_path, results: Dict[str, pd.DataFrame], survival: bool) -> Dict:
    """
    Return the current results as dataframes. It creates a CSV file with the pairs and its values. Keeping in
    mind that the results are pairs it uses the suffixes ``_a`` and ``_b`` to denote each member of the pair

        - ``time_a``: Survival time of pair's member A
        - ``time_b``: Survival time of pair's member B
        - ``pairs_a``: Key of pair's member A
        - ``pairs_b``: Key of pair's member B
        - ``labels``: Labels that are true if :math:`T(p_a) < T(p_b)`
        - ``predictions``: Predictions made by the current model
    :
    """
    # Load clinical info
    target_df = pd.read_csv(target_path, index_col=0)
    merged = {}
    for name, result in results.items():
        if survival:
            merged[name] = _select_time(target_df, result)
        else:
            merged[name] = _select_target(target_df, result)
    return merged


def save_cox_results(results: Dict[str, pd.DataFrame], path: str):
    """
    Save the cox results to disk. It creates a CSV file with the pairs and its values. Keeping in
    mind that the results are pairs it uses the suffixes ``_a`` and ``_b`` to denote each member of the pair

    """

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for name, result in results.items():
        result = result[['pA', 'pB', 'distance', 'predict_a', 'predict_b', 'time_a', 'time_b', 'comp', 'predict_comp']]
        result.to_csv(os.path.join(path, f"{name}_results.csv"))


