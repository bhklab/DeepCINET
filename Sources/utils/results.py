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


def save_results(sess: tf.Session,
                 results: Dict[str, pd.DataFrame],
                 path: str,
                 save_model: bool,
                 survival: bool,
                 target_path: str):
    """
    Save the current results to disk. It creates a CSV file with the pairs and its values. Keeping in
    mind that the results are pairs it uses the suffixes ``_a`` and ``_b`` to denote each member of the pair
        - ``target_value_a``: Survival time/drug sensitivity of pair's member A
        - ``target_value_b``: Survival time/drug sensitivity of pair's member B
        - ``pairs_a``: Key of pair's member A
        - ``pairs_b``: Key of pair's member B
        - ``labels``: Labels that are true if :math:`T(p_a) < T(p_b)`
        - ``predictions``: Predictions made by the current model
        - ``predicted_value_a``: Predicted value for the pair's member A
        - ``predicted_value_b``: Predicted value for the pair's member B

    Moreover, the model is also saved into disk. It can be found in the ``path/weights/`` directory and can
    loaded with Tensorflow using the following commands:

    >>> import tensorflow as tf
    >>> saver = tf.train.Saver()
    >>> with tf.Session() as sess:
    >>>     saver.restore(sess, "<path>/weights/weights.ckpt")

    :param sess: Current session that should be saved when saving the model
    :param results: List with tuples with a name and a :class:`pandas.DataFrame` of results that should be saved.
                    the :class:`pandas.DataFrame` should contain at least the columns
                    ``pairs_a``, ``pairs_b``, ``labels`` and ``predictions``.
    :param path: Directory path where all the results should be saved
    :param save_model: If :any:`True` save the model to disk
    :param survival: It specify that whether the results are related to the survival prediction or not
    :param target_path: This is showing the path to the target dataframe
    """
    weights_dir = os.path.join(path, 'weights')

    # Always overwrite the previous weights
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(weights_dir)

    if save_model:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(weights_dir, 'weights.ckpt'))

    # Load clinical info
    clinical_info = pd.read_csv(target_path, index_col=0)

    for name, result in results.items():
        if survival:
            merged = _select_time(clinical_info, result)
        else:
            merged = _select_target(clinical_info, result)
        merged.to_csv(os.path.join(path, f"{name}_results.csv"))


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
    merge = merge.rename(index=str, columns={'target': 'target_value_b',
                                             'target_a': 'target_value_a',
                                             'gather_a':'predict_value_a',
                                             'gather_b':'predict_value_b'})
    merge['real_dist'] = merge['target_value_b'] - merge['target_value_a']
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
