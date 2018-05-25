"""
Module to work with result data
"""
import glob
import collections
import logging

import pandas as pd
import numpy as np

import settings

clinical = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED)
print(clinical.columns)


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
    logger = logging.getLogger(__name__)

    logger.debug(f"Searching on {path} {select_type}")
    files = glob.glob(path + f"/fold*/{select_type}*.csv")
    logger.debug(f"Found {len(files)}")

    df_list = []
    elem_comparisons = {}
    for file in files:
        df = pd.read_csv(file, index_col=0)

        elem_right = len(df[df["labels"] == df["predictions"]])
        elem_count = len(df)

        ids = np.concatenate((df['pA'].values, df['pB'].values))
        ids = collections.Counter(ids)
        key, count = ids.most_common(1)[0]

        is_censored = not clinical.loc[clinical['id'] == key, 'event'].values[0]

        df_list.append(pd.DataFrame({
            "id": [key],
            "right": [elem_right],
            "total": [elem_count],
            "censored": [is_censored]
        }))

        if elem_folds or predictions:
            elem_comparisons[key] = df

    if predictions:
        predictions_df: pd.DataFrame = pd.concat(elem_comparisons.values(), ignore_index=True)
    else:
        predictions_df = None

    results_df: pd.DataFrame = pd.concat(df_list, ignore_index=True)
    if select_type == "mixed":
        results_df['c-index'] = results_df['right']/results_df['total']

    no_cens_results = results_df.loc[~results_df['censored']]

    logger.info(f"Finished {path} {select_type}\n")
    return (results_df, no_cens_results), predictions_df, elem_comparisons


