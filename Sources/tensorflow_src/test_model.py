import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import pathlib


import tensorflow as tf
import pandas as pd

import data
from data.train_test import get_sets_generator, get_sets_reader
import tensorflow_src.models as models
import tensorflow_src.models.basics
import tensorflow_src.config as settings
import tensorflow_src.train_test_models as train_test_models
import utils



def test_DeepCinet(weights_dir: str ,
                   data_type: str = 'clinical',
                   results_path: str = settings.SESSION_TEST_SAVE_PATH,
                   log_device=False,
                   siamese_model="",
                   batch_size=""

             ):
    """
    deepCient
    :param feature_list:
    :param args: Command Line Arguments
    """
    #tf.reset_default_graph()
    results_path = pathlib.Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")

    logger.info(f"Results path: {results_path}")

    logger.info("Script to test a siamese neural network model")
    logger.info(f"NetworkPath:{model_path}")
    features = pd.DataFrame()
    # read features and clinical data frame the path is defined in the settings.py
    logger.info(f"data type: {data_type}")
    if data_type == "radiomic":
        features = pd.read_csv(settings.DATA_TEST_PATH_RADIOMIC_PROCESSED, index_col=0)
    elif data_type == "clinical":
        features = pd.read_csv(settings.DATA_TEST_PATH_CLINIC_PROCESSED, index_col=0)
    elif data_type == "clinicalVolume":
        features = pd.read_csv(settings.DATA_TEST_PATH_VOLUME_CLINIC_PROCESSED, index_col=0)

    logger.info(f"number of features is {len(features.index)}")
    clinical_df = pd.read_csv(settings.DATA_TEST_PATH_CLINICAL_PROCESSED, index_col=0)
    logger.info("read Feature DataFrame")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(weights_dir, 'weights.ckpt'))

        counts = {}
        for key in ['test', 'mixed']:
            counts[key] = {
                'total': 0,
                'correct': 0,
                'c_index': []
            }
        dataset = data.pair_data.SplitPairs()
        clinical_data = dataset.clinical_data.copy()
        df_features = features
        batch_data = data.BatchData(df_features)

        train_pairs, test_pairs, mixed_pairs = dataset.create_train_test(train_data, test_data,
                                                                                 random=False)
         # Initialize all the variables
        logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
        predictions = {}
        for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
            if len(pairs) <= 0:
                continue
            logger.info(f"Computing {name} c-index")
            correct, total, results = \
                train_test_models.test_iterations(sess,
                                siamese_model,
                                batch_data,
                                pairs,
                                batch_size)

            correct = int(correct)

            c_index = correct / total

            counts[name]['total'] += total
            counts[name]['correct'] += correct
            counts[name]['c_index'].append((0, c_index))

            predictions[name] = results

            logger.info(f"{name} set c-index: {c_index}, correct: {correct}, total: {total}, "
                        f"temp c-index: {counts[name]['correct']/counts[name]['total']}")



            # Save each fold in a different directory


        logger.info(f"Saving results at: {results_save_path}")
        utils.save_results(sess, predictions, results_save_path, False)
        logger.info("\r ")

        for key in counts:
            if counts[key]['total'] <= 0:
                continue
            logger.info(f"Final {key} c-index: {counts[key]['correct']/counts[key]['total']}")
        return counts, predictions



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # parser._action_groups.pop()

    required = parser.add_argument_group('required named arguments')
    optional = parser.add_argument_group('optional named arguments')

    # Required arguments
    required.add_argument(
        "--model",
        help="Choose the model that you want to use for training",
        type=str,
        required=True
    )

    # Optional arguments
    optional.add_argument(
        "-h", "--help",
        help="Show this help",
        action="help"
    )

    required.add_argument(
        "--trained-network",
        help="The location of the trained network"
             "Note that trained network should be trained on the same data-set as the test data which you want to used",
        type=int
    )
    optional.add_argument(
        "--test-path",
        help="path to the test data",
        default=.25,
        type=float
    )
    optional.add_argument(
        "--gpu-level",
        help="Amount of GPU resources used when fitting the model. 0: no GPU usage, "
             "1: only second conv layers, 2: all conv layers, "
             "3: all layers and parameters are on the GPU",
        default=0,
        type=int
    )
    optional.add_argument(
        "--gpu-allow-growth",
        help="Allow Tensorflow to use dynamic allocations with GPU memory",
        default=False,
        action="store_true",
    )
    optional.add_argument(
        "--results-path",
        help="Path where the results and the model should be saved",
        default=settings.SESSION_SAVE_PATH,
        type=str
    )
    optional.add_argument(
        "--data_type",
        help="the type of features",
        default="radiomic",
        type=str
    )
    optional.add_argument(
        "--log-device",
        help="Log device placement when creating all the tensorflow tensors",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--full-summary",
        help="Write a full summary for tensorboard, otherwise only the scalar variables will be logged",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--save-model",
        help="Save the model to the location specified at the results path",
        action="store_true",
        default=False
    )

    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

    arguments['results_path'] = os.path.abspath(arguments['results_path'])
    results_path = pathlib.Path(arguments['results_path'])
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{array_id}', str(results_path))

    logger.debug("Script starts")
    logger.debug(arguments)
    logger.info(f"Results path: {results_path}")

    if len(unknown) > 0:
        logger.warning(f"Warning: there are unknown arguments {unknown}")

    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
