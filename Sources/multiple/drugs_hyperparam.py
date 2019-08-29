#!/usr/bin/env python3
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import tensorflow_src.config as config
import utils
# import STprediction
from tensorflow_src import gpu_hyperParamSelection
import random
import yaml

path = config.DATA_PATH_PROCESSED
if __name__ == '__main__':
    logger = utils.init_logger("drugs_pyperparam")
    for drug in ['lapatinib']:
        results_path = f"Result/{drug}"
        target_path = f"drug_response/drug_CTRPv2_{drug}.csv"
        feature_path = f"L1000_gene_expression/gene_CTRPv2_{drug}.csv"
        input_path = f"train_test_{drug}"
        logger = utils.init_logger("multiple run random")
        try:
            gpu_hyperParamSelection.hyperParamSelection(results_path=results_path,
                                                        target_path=target_path,
                                                        feature_path=feature_path,
                                                        input_path=input_path)
            logger.info(f"don hyper pram for drug {drug}")
        except:
            logger.info(f"somethin is wrong for drug {drug}", sys.exc_info())
